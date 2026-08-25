import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import datetime
import time
import json
import os

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(page_title="조달계약 검토 도우미", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
[data-testid="stChatMessageAvatarUser"] {
    background-color: #2563eb !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background-color: #16a34a !important;
}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background-color: #6b7280 !important;
    border-color: #6b7280 !important;
    color: #ffffff !important;
}
[data-testid="stChatInput"],
[data-testid="stChatInput"]:hover,
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] textarea:hover,
[data-testid="stChatInput"] textarea:focus {
    border-color: #9ca3af !important;
    box-shadow: none !important;
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("Gemini API 키가 설정되지 않았습니다. Streamlit Cloud의 Settings → Secrets에 "
             "GEMINI_API_KEY 를 추가해주세요. (README.md 참고)")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# 유료 티어에서는 무료 한도 걱정이 없으므로, 성능이 가장 좋은 모델을 우선순위로 둔다.
# 맨 뒤의 lite 모델들은 혹시 모를 일시적 오류에 대비한 안전망일 뿐이다.
MODEL_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# 답변할 때 구글 검색으로 근거를 보완할 수 있게 하는 도구
SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

MAX_DOCS = 15  # 무료 서버 메모리 한도를 고려한 지식창고 문서 개수 상한

SYSTEM_INSTRUCTION = """너는 공기업 계약부서의 AI 계약 검토 보조자 '조달계약 검토 도우미'다.
참고자료(지식창고)와 사용자가 첨부한 문서를 근거로 답한다.

입력을 받으면 먼저 아래 4가지 중 어떤 유형인지 스스로 판단한다.
파일 형식이 A·B·C처럼 보이더라도, 사용자의 질문/요청 내용이 해당 유형의
검토 목적과 맞지 않으면 D(자유 질문)로 판단하고 D의 방식으로 답한다.

[A] 과업지시서 / 구매규격서 / 특수조건 등 계약 관련 문서 검토 요청
    → 아래 [검토 리포트 출력 순서 및 양식]을 그대로 따라 작성한다.

[검토 리포트 출력 순서 및 양식]
## 1. 📄 문서 개요
- **건명**:
- **발주/작성부서**:
- **주요 과업 내용**: (2~3줄로 간결하게 요약)
---
## 2. 🚨 주요 위반 및 독소조항 검토 (우선 검토)
* 독소조항, 부당특약, 특정 사양 명시, 법령 위반 등 수정/보완이 필요한 항목을
  심각도별로 우선 출력한다. (문제 항목이 없다면 "특이 위반 사항 없음"으로 명시)
* 심각도 구분:
  - 🔴 [수정 필요]: 법령 위반, 부당한 책임 전가, 특정 사양 지정 등 계약 체결 전
    반드시 수정해야 할 항목
  - 🟡 [검토 권고]: 사유서 첨부 필요, 경쟁 제한 소지, 문구 명확화가 필요한 항목
* 작성 형식:
  - **위치/조항**: (예: 규격서 5.1.1항)
  - **현행 문구**: "..."
  - **문제점 및 사유**: (구체적 문제 사유)
  - **💡 AI 수정 권고안**: "..."
  - **관련 근거**: (지식창고 내 법령/규정 조항 명시)
---
## 3. ⚖️ 계약방법 적정성 및 진행 프로세스
- **계약방법 적정성**: (지정된 계약방식의 타당성 및 수의계약/특정규격 사유서 필요 여부)
- **향후 진행 절차**: (보완 후 입찰공고 및 계약 체결까지의 필수 단계를 순서대로 안내)
---
## 4. 🟢 정상(적정) 검토 항목
- 문제없이 법령 및 사내 규정에 맞게 잘 설계된 주요 항목 정리

[작성 주의사항]
- "단정할 수 없습니다"와 같은 방어적 면책 표현은 지양하고, 지식창고 내 가장
  연관성 높은 조항을 인용하여 객관적 검토 의견을 제시할 것. (다만 지식창고와
  인터넷 검색 모두에 근거가 전혀 없는 경우는 예외로, 그 사실을 명확히 알린다.)
- 담당자가 결재 및 내부 보고용으로 바로 활용할 수 있도록 정돈된 마크다운
  양식으로 작성할 것.

[B] 산출내역서 / 원가계산서(수치·요율 포함 문서) 검토 요청
    → 1) 문서 요약  2) 항목별 요율 검증(기준 대비 초과/적정 여부)
      3) 근거 규정  4) 수정이 필요한 항목 요약

[C] 사유서(수의계약 사유서, 특정규격 지정 사유서 등) 검토 요청
    → 1) 적용 사유 조항 확인  2) 요건 충족 여부
      3) 누락된 첨부서류 체크  4) 보완 필요 사항

[D] 위 세 가지 검토 요청이 아닌 모든 경우
    (계약방법 문의, 프로세스 질문, 이어지는 대화, 첨부파일은 A/B/C 문서지만
     질문이 단순 확인·설명 요청 등 검토 목적이 아닌 경우 포함)
    → 정해진 형식 없이, 일반 LLM처럼 질문 의도에 맞게 자유롭게 답한다.

법령/규정의 근거가 필요한데 참고자료(지식창고)에서 찾지 못했다면,
Google 검색 도구를 사용해 인터넷에서 최신 법령·개정 내용을 찾아본다.
이 경우 답변 안에 "(인터넷 검색 근거: OOOO년 OO월 개정 OOO법 제O조 참고)"
형식으로 반드시 표시하고, 지식창고 근거와 인터넷 검색 근거를 구분해서 안내한다.
지식창고에도 인터넷 검색에도 근거가 없으면 "근거를 찾지 못해 단정할 수 없습니다"라고 명시한다.
이후 사용자가 이어서 질문하면 이전 대화 맥락을 유지하며 답한다."""

# =========================================================
# 디스크 저장 (새로고침해도 유지되도록)
# =========================================================
DATA_FILE = "app_data.json"


def save_state():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "knowledge_base": st.session_state.get("knowledge_base", {}),
                "conversations": st.session_state.get("conversations", {}),
                "current_conv": st.session_state.get("current_conv"),
                "last_refreshed": st.session_state.get("last_refreshed"),
            }, f, ensure_ascii=False)
    except Exception:
        pass  # 저장 실패해도 앱은 계속 동작하게 둔다


def load_state():
    if not os.path.exists(DATA_FILE):
        return
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.knowledge_base = data.get("knowledge_base", {})
        st.session_state.conversations = data.get("conversations", {})
        if data.get("current_conv") in st.session_state.conversations:
            st.session_state.current_conv = data["current_conv"]
        if data.get("last_refreshed"):
            st.session_state.last_refreshed = data["last_refreshed"]
    except Exception:
        pass


# =========================================================
# 세션 상태 초기화
# =========================================================
if "_state_loaded" not in st.session_state:
    load_state()
    st.session_state._state_loaded = True

if "conversations" not in st.session_state:
    # {conv_id: {"title":..., "messages":[{"role":"user"/"model","text":...}]}}
    st.session_state.conversations = {}
if "current_conv" not in st.session_state:
    st.session_state.current_conv = None
if "knowledge_base" not in st.session_state:
    # {문서명: {"text":..., "uploaded_at":...}}
    st.session_state.knowledge_base = {}


def new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": "새 대화", "messages": []}
    st.session_state.current_conv = conv_id
    save_state()


if st.session_state.current_conv is None:
    new_conversation()


# =========================================================
# 문서 처리 (지식창고)
# =========================================================
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        return file.read().decode("utf-8", errors="ignore")


def chunk_text(text, size=800, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return [c for c in chunks if c.strip()]


def add_document(file):
    if file.name not in st.session_state.knowledge_base and len(st.session_state.knowledge_base) >= MAX_DOCS:
        st.warning(f"지식창고는 최대 {MAX_DOCS}개 문서까지 권장합니다. 안 쓰는 문서를 먼저 삭제해주세요.")
        return
    text = extract_text(file)
    st.session_state.knowledge_base[file.name] = {
        "text": text,
        "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_state()


def refresh_all_documents():
    """지식창고에 있는 모든 문서를 한 번에 최신 버전으로 교체하는 자리.
    현재는 새로고침 시각만 기록하며, 법제처 Open API 키를 발급받으면
    이 함수 안에서 문서별로 API를 호출해 공포일자를 비교하고 자동 교체하도록 확장할 수 있다."""
    st.session_state.last_refreshed = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    save_state()


def search_relevant_chunks(query, top_k=4):
    all_chunks = []
    sources = []
    for name, doc in st.session_state.knowledge_base.items():
        for c in chunk_text(doc["text"]):
            all_chunks.append(c)
            sources.append(name)
    if not all_chunks:
        return []
    vectorizer = TfidfVectorizer().fit(all_chunks + [query])
    chunk_vecs = vectorizer.transform(all_chunks)
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, chunk_vecs)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    return [(sources[i], all_chunks[i]) for i in top_idx if sims[i] > 0]


# =========================================================
# 좌측 사이드바: 파일 목록(상단) + 새 대화/이력(하단)
# =========================================================
with st.sidebar:
    st.markdown("### ⚖️ 조달계약 검토 도우미")

    header_col, refresh_col = st.columns([5, 1])
    with header_col:
        st.markdown("#### 📎 지식창고 (법령·규정)")
    with refresh_col:
        if st.button("🔄", key="refresh_all", help="전체 문서 최신본으로 새로고침"):
            refresh_all_documents()

    uploaded = st.file_uploader("법령/규정/표준품셈 업로드", type=["pdf", "txt"],
                                 accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        for f in uploaded:
            add_document(f)
        st.success(f"{len(uploaded)}개 문서 반영 완료")

    with st.expander(f"반영된 문서 보기 ({len(st.session_state.knowledge_base)}개)", expanded=False):
        with st.container(height=300):
            for name in list(st.session_state.knowledge_base):
                c1, c2 = st.columns([5, 1])
                c1.caption(f"📄 {name}")
                if c2.button("🗑", key=f"del_{name}", help="이 문서 삭제"):
                    del st.session_state.knowledge_base[name]
                    save_state()
                    st.rerun()

    if "last_refreshed" in st.session_state:
        st.caption(f"업데이트 완료: {st.session_state.last_refreshed}")

    st.divider()

    if st.button("➕ 새 대화", use_container_width=True):
        new_conversation()

    st.markdown("#### 대화 이력")
    for conv_id, conv in reversed(list(st.session_state.conversations.items())):
        label = conv["title"]
        is_active = conv_id == st.session_state.current_conv
        if st.button(label, key=f"conv_{conv_id}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.current_conv = conv_id
            save_state()
            st.rerun()

# =========================================================
# 메인 화면
# =========================================================
current = st.session_state.conversations[st.session_state.current_conv]

if not current["messages"]:
    st.markdown("### 검토할 문서를 올리거나 상황을 입력하세요")
    st.caption("좌측의 법령·규정 내용을 근거로 검토해드립니다.")


def call_ai(context_messages, user_text, file_text):
    """context_messages 이전까지의 대화 이력을 바탕으로, 이번 사용자 입력에 대한 AI 답변을 만든다.
    후보 모델을 순서대로 시도하고, 다 실패하면 어떤 모델이 왜 실패했는지 진단 정보를 함께 보여준다."""
    search_query = (user_text + "\n" + file_text).strip()
    relevant = search_relevant_chunks(search_query)
    context_text = "\n\n---\n\n".join(f"[{src}]\n{chunk}" for src, chunk in relevant)

    history = []
    for m in context_messages:
        role = "user" if m["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=m["text"])]))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[SEARCH_TOOL],
    )
    prompt = (
        f"[참고자료(지식창고)]\n{context_text if context_text else '(지식창고에 문서 없음)'}\n\n"
        f"[이번에 첨부된 문서]\n{file_text if file_text else '(첨부 없음)'}\n\n"
        f"[사용자 질문/요청]\n{user_text if user_text else '(첨부 문서를 검토해줘)'}"
    )

    attempts = []  # (model_name, 짧은 오류 요약) 기록 — 진단용
    for idx, model_name in enumerate(MODEL_CANDIDATES):
        try:
            chat = client.chats.create(model=model_name, config=config, history=history)
            response = chat.send_message(prompt)
            return response.text  # 성공하면 바로 반환
        except Exception as e:
            error_str = str(e)
            is_quota = "RESOURCE_EXHAUSTED" in error_str or "429" in error_str
            attempts.append((model_name, "한도 초과(429)" if is_quota else error_str[:150]))
            if is_quota and idx < len(MODEL_CANDIDATES) - 1:
                time.sleep(1)  # 순간적으로 몰려서 같이 걸리는 걸 막기 위한 짧은 대기
                continue
            if not is_quota:
                break  # 한도 문제가 아니면 나머지 후보를 더 시도해도 의미가 없어 중단

    detail = "\n".join(f"- {name}: {err}" for name, err in attempts)
    return (
        "⚠️ 시도한 모델이 모두 실패했습니다. 아래는 각 모델별 원인입니다:\n\n"
        f"{detail}\n\n"
        "전부 '한도 초과'라면 https://ai.dev/rate-limit 에서 실제 사용량을 다시 확인해보시고, "
        "그 외 오류가 섞여 있다면 해당 모델 이름 자체에 문제가 있을 수 있습니다."
    )


def run_new_turn(current, user_text, file_text, file_name):
    if user_text and file_name:
        display_text = f"📎 {file_name}\n\n{user_text}"
    elif file_name:
        display_text = f"📎 {file_name}"
    else:
        display_text = user_text

    context_messages = list(current["messages"])  # 이 시점까지가 이전 대화 이력
    current["messages"].append({
        "role": "user", "text": display_text,
        "raw_text": user_text, "file_text": file_text, "file_name": file_name,
    })
    if current["title"] == "새 대화":
        title_source = user_text if user_text else (file_name or "새 대화")
        current["title"] = title_source[:20] + ("..." if len(title_source) > 20 else "")

    answer_text = call_ai(context_messages, user_text, file_text)
    current["messages"].append({"role": "model", "text": answer_text})
    save_state()


# 대화 내역 출력 + 질문/답변 쌍 삭제 기능
i = 0
while i < len(current["messages"]):
    msg = current["messages"][i]

    if msg["role"] == "user":
        col1, col2 = st.columns([10, 1])
        with col1:
            with st.chat_message("user"):
                st.markdown(msg["text"])
        with col2:
            if st.button("🗑️", key=f"del_pair_{i}", help="이 질문과 답변 삭제"):
                current["messages"].pop(i)
                if i < len(current["messages"]) and current["messages"][i]["role"] == "model":
                    current["messages"].pop(i)
                if not current["messages"]:
                    current["title"] = "새 대화"
                save_state()
                st.rerun()
        i += 1
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["text"])
        i += 1

chat_value = st.chat_input(
    "문서를 첨부하거나 질문을 입력하세요",
    accept_file=True,
    file_type=["pdf", "txt"],
)

if chat_value:
    user_text = (chat_value.text or "").strip()
    attached_file = chat_value.files[0] if chat_value.files else None
    file_text = extract_text(attached_file) if attached_file is not None else ""
    file_name = attached_file.name if attached_file is not None else None

    run_new_turn(current, user_text, file_text, file_name)
    st.rerun()
