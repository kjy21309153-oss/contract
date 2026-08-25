import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import datetime
import re
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

</style>
""", unsafe_allow_html=True)

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("Gemini API 키가 설정되지 않았습니다. Streamlit Cloud의 Settings → Secrets에 "
             "GEMINI_API_KEY 를 추가해주세요. (README.md 참고)")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


@st.cache_resource
def get_working_model_name():
    """구글이 모델 이름을 바꾸거나 없애도 앱이 안 죽도록,
    여러 후보 이름 중 실제로 쓸 수 있는 걸 자동으로 찾는다."""
    candidates = ["gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                    "gemini-2.5-flash-lite", "gemini-2.5-flash"]
    try:
        available = {m.name.split("/")[-1] for m in client.models.list()}
        for name in candidates:
            if name in available:
                return name
    except Exception:
        pass
    # 목록 조회에 실패하면 첫 번째 후보로 일단 시도
    return candidates[0]


MODEL_NAME = get_working_model_name()

# 답변할 때 구글 검색으로 근거를 보완할 수 있게 하는 도구
SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

MAX_DOCS = 15  # 무료 서버 메모리 한도를 고려한 지식창고 문서 개수 상한

SYSTEM_INSTRUCTION = """너는 공기업 계약부서의 AI 계약 검토 보조자 '조달계약 검토 도우미'다.
참고자료(지식창고)와 사용자가 첨부한 문서를 근거로 답한다.

입력을 받으면 먼저 아래 4가지 중 어떤 유형인지 스스로 판단한다.
파일 형식이 A·B·C처럼 보이더라도, 사용자의 질문/요청 내용이 해당 유형의
검토 목적과 맞지 않으면 D(자유 질문)로 판단하고 D의 방식으로 답한다.

[A] 과업지시서 / 구매규격서 / 특수조건 등 계약 관련 문서 검토 요청
    → 1) 문서 요약  2) 계약방법 적정성(왜 이 방법이 맞는지)
      3) 법령·규정 검토(문제없는 항목도 이유와 함께, 문제있는 항목은 근거조문+수정안)
      4) 문제 해결 후 진행 프로세스

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
    st.caption("좌측에서 법령·규정을 먼저 올려두면, 그 내용을 근거로 검토해드립니다.")

MODEL_CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                    "gemini-2.5-flash-lite", "gemini-2.5-flash"]

def call_ai(context_messages, user_text, file_text):
    """context_messages 이전까지의 대화 이력을 바탕으로, 이번 사용자 입력에 대한 AI 답변을 만든다."""
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

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            chat = client.chats.create(model=model_name, config=config, history=history)
            response = chat.send_message(prompt)
            return response.text  # 성공하면 그 자리에서 바로 반환 (다음 후보 시도 안 함)
        except Exception as e:
            error_str = str(e)
            last_error = e
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                continue  # 이 모델만 막혔으니 다음 후보로 넘어감
            else:
                break  # 한도 문제가 아닌 다른 오류면 재시도해도 의미 없으니 중단

    # 후보를 다 돌았는데도 실패한 경우
    return f"⚠️ 지금 사용 가능한 모델이 모두 한도를 초과했습니다. 잠시 후 다시 시도해주세요. ({last_error})"


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


for msg in current["messages"]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["text"])

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
