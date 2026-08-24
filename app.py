import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import datetime
import re

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(page_title="계약이 Ver.3", page_icon="⚖️", layout="wide")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("Gemini API 키가 설정되지 않았습니다. Streamlit Cloud의 Settings → Secrets에 "
             "GEMINI_API_KEY 를 추가해주세요. (README.md 참고)")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
# 무료 티어에서 사용 가능한 모델 (2026년 기준 Flash 계열). 만료/변경 시 아래 이름만 바꾸면 됩니다.
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

SYSTEM_INSTRUCTION = """너는 공기업 계약부서의 AI 계약 검토 보조자 '계약이'다.
사용자가 올린 법령/규정/표준품셈 등 참고자료(지식창고)를 근거로만 답한다.
문서를 검토할 때는 아래 형식을 따른다:
1. 문서 요약 (무슨 계약 건인지)
2. 계약방법 적정성 (왜 이 계약방법이 맞는지, 근거)
3. 법령/규정 검토 (문제 없는 항목도 이유와 함께, 문제 있는 항목은 근거 조문과 수정안)
4. 문제 해결 후 진행 프로세스 (계약방법에 따른 다음 단계)
참고자료에 없는 내용은 "지식창고에 근거자료가 없어 단정할 수 없습니다"라고 명시한다.
이후 사용자가 이어서 질문하면 이전 대화 맥락을 유지하며 답한다."""

# =========================================================
# 세션 상태 초기화
# =========================================================
if "conversations" not in st.session_state:
    # {conv_id: {"title":..., "messages":[{"role":"user"/"model","text":...}]}}
    st.session_state.conversations = {}
if "current_conv" not in st.session_state:
    st.session_state.current_conv = None
if "knowledge_base" not in st.session_state:
    # {문서명: {"text":..., "uploaded_at":..., "chunks":[...]}}
    st.session_state.knowledge_base = {}


def new_conversation():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": "새 대화", "messages": []}
    st.session_state.current_conv = conv_id


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
    text = extract_text(file)
    st.session_state.knowledge_base[file.name] = {
        "text": text,
        "uploaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "chunks": chunk_text(text),
    }


def refresh_document(name):
    """같은 이름의 문서를 최신 버전으로 교체하는 자리.
    현재는 재업로드를 안내하는 형태이며, 법제처 Open API 키를 발급받으면
    이 함수 안에서 API를 호출해 공포일자를 비교하고 자동 교체하도록 확장할 수 있다."""
    st.session_state.pop("refresh_target", None)
    st.info(f"'{name}' 최신본을 다시 업로드해주세요. (자동 API 연동은 README의 "
            f"'법령 자동 업데이트 붙이기' 참고)")


def search_relevant_chunks(query, top_k=4):
    all_chunks = []
    sources = []
    for name, doc in st.session_state.knowledge_base.items():
        for c in doc["chunks"]:
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
    st.markdown("### ⚖️ 계약이 Ver.3")

    st.markdown("#### 📎 지식창고 (법령·규정)")
    uploaded = st.file_uploader("법령/규정/표준품셈 업로드", type=["pdf", "txt"],
                                 accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        for f in uploaded:
            add_document(f)
        st.success(f"{len(uploaded)}개 문서 반영 완료")

    for name, doc in st.session_state.knowledge_base.items():
        c1, c2 = st.columns([4, 1])
        c1.caption(f"📄 {name}  \n`{doc['uploaded_at']}`")
        if c2.button("🔄", key=f"refresh_{name}", help="최신본으로 새로고침"):
            refresh_document(name)

    st.divider()

    if st.button("➕ 새 대화", use_container_width=True):
        new_conversation()

    st.markdown("#### 대화 이력")
    for conv_id, conv in reversed(list(st.session_state.conversations.items())):
        label = conv["title"]
        if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
            st.session_state.current_conv = conv_id

# =========================================================
# 메인 화면
# =========================================================
current = st.session_state.conversations[st.session_state.current_conv]

if not current["messages"]:
    st.markdown("### 검토할 문서를 올리거나 상황을 입력하세요")
    st.caption("좌측에서 법령·규정을 먼저 올려두면, 그 내용을 근거로 검토해드립니다.")
else:
    for msg in current["messages"]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["text"])

user_input = st.chat_input("문서 내용을 붙여넣거나 질문을 입력하세요")

if user_input:
    current["messages"].append({"role": "user", "text": user_input})
    if current["title"] == "새 대화":
        current["title"] = user_input[:20] + ("..." if len(user_input) > 20 else "")

    with st.chat_message("user"):
        st.markdown(user_input)

    relevant = search_relevant_chunks(user_input)
    context_text = "\n\n---\n\n".join(f"[{src}]\n{chunk}" for src, chunk in relevant)

    history = []
    for m in current["messages"][:-1]:
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["text"]]})

    chat = model.start_chat(history=history)
    prompt = f"{SYSTEM_INSTRUCTION}\n\n[참고자료]\n{context_text if context_text else '(지식창고에 문서 없음)'}\n\n[질문]\n{user_input}"

    with st.chat_message("assistant"):
        with st.spinner("검토 중..."):
            response = chat.send_message(prompt)
            st.markdown(response.text)

    current["messages"].append({"role": "model", "text": response.text})
    st.rerun()
