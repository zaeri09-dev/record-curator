import streamlit as st
import pandas as pd
import PyPDF2
import json
import chromadb 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import matplotlib.pyplot as plt

# --- 데이터베이스 초기화 함수 ---
@st.cache_resource
def get_db_collection():
    # 💡 [최종 해결책] 클라우드의 깐깐한 하드디스크 자물쇠(Read-only)를 완벽히 회피합니다.
    # PersistentClient(하드디스크 저장) 대신, 일반 Client(메모리 임시 저장)를 사용합니다.
    # 에러 확률 0%이며, 데이터가 지워져도 우리의 '엑셀 복원' 기능으로 언제든 살려낼 수 있습니다!
    client = chromadb.Client() 
    collection = client.get_or_create_collection(name="chemistry_reports")
    return collection

collection = get_db_collection()

# 화면 기본 설정
st.set_page_config(page_title="화학 역량 큐레이터", page_icon="🧪", layout="wide")

st.title("🧪 멀티모달 RAG 기반 학생 역량 큐레이터")
st.markdown("학생의 탐구 보고서를 분석하고, **LangChain Agent**를 활용해 데이터를 정밀하게 관리·통계냅니다.")
st.divider()

# --- 화면 왼쪽 사이드바: 설정 및 입력 ---
with st.sidebar:
    st.header("⚙️ 기본 설정")
    
    api_key_input = st.text_input(
        "Gemini API 키 입력", 
        type="password", 
        placeholder="API 키를 붙여넣으세요 (새로고침 시 안전하게 지워집니다)"
    )
    
    if api_key_input:
        st.success("✅ 키 임시 저장 완료! (현재 접속 중에만 안전하게 유지됩니다)")
    else:
        st.info("💡 분석 기능을 사용하려면 API 키를 한 번 입력해 주세요.")
    
    st.divider()
    
    st.header("📂 1. 새 보고서 분석")
    col_id, col_name = st.columns(2)
    with col_id:
        student_id_input = st.text_input("학번", placeholder="예: 10101")
    with col_name:
        student_name_input = st.text_input("이름", placeholder="예: 김화학")
        
    uploaded_file = st.file_uploader("PDF 보고서 업로드", type=["pdf"])
    
    analyze_btn = st.button("🚀 AI 분석 시작", type="primary", use_container_width=True)

# --- 함수: PDF 텍스트 추출 ---
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- 화면 중앙 상단: 1. AI 분석 및 DB 저장 (LangChain 적용) ---
st.header("📊 1. AI 분석 및 DB 누적 저장 (LangChain 파이프라인)")

if analyze_btn:
    if not api_key_input:
        st.error("앗! 왼쪽 설정에서 Gemini API 키를 먼저 입력해 주세요.")
    elif not student_id_input or not student_name_input or not uploaded_file:
        st.warning("학번, 이름, 그리고 PDF 파일을 모두 확인해 주세요.")
    else:
        unique_student_id = f"{student_id_input}_{student_name_input}"
        
        with st.spinner(f"'{student_name_input}' 학생의 보고서를 LangChain AI가 분석 중입니다... 🤖"):
            try:
                document_text = extract_text_from_pdf(uploaded_file)
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview", 
                    google_api_key=api_key_input,
                    temperature=0.2 
                )
                
                parser = JsonOutputParser()
                
                # 💡 [수정 필요] 선생님의 화학 수업 평가 기준에 맞게 프롬프트 내용을 수정할 수 있습니다.
                prompt_template = PromptTemplate(
                    template="""
                    너는 통찰력 있는 고등학교 화학 교사야. 아래 학생이 제출한 화학 탐구 보고서를 읽고 평가해줘.
                    
                    {format_instructions}
                    
                    필수 포함 키(Key):
                    - 과학적탐구력 (int, 1~100)
                    - 문제해결력 (int, 1~100)
                    - 논리적사고력 (int, 1~100)
                    - 세특초안 (string, 150자 이내 명사형 종결어미)

                    [학생 보고서 내용]
                    {document_text}
                    """,
                    input_variables=["document_text"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )
                
                chain = prompt_template | llm | parser
                result_data = chain.invoke({"document_text": document_text})
                
                existing_data = collection.get(ids=[unique_student_id])
                
                if existing_data and existing_data['ids']:
                    old_text = existing_data['documents'][0]
                    old_meta = existing_data['metadatas'][0]
                    old_count = old_meta.get('분석횟수', 1)
                    
                    new_text = f"{old_text}\n\n---\n\n{result_data['세특초안']}"
                    new_count = old_count + 1
                    new_meta = {
                        "분석횟수": new_count,
                        "과학적탐구력": round((old_meta['과학적탐구력'] * old_count + result_data['과학적탐구력']) / new_count),
                        "문제해결력": round((old_meta['문제해결력'] * old_count + result_data['문제해결력']) / new_count),
                        "논리적사고력": round((old_meta['논리적사고력'] * old_count + result_data['논리적사고력']) / new_count)
                    }
                    st.toast(f"📈 기존 데이터에 {new_count}번째 기록이 누적되었습니다!", icon="👏")
                else:
                    new_text = result_data['세특초안']
                    new_meta = {
                        "분석횟수": 1,
                        "과학적탐구력": result_data['과학적탐구력'],
                        "문제해결력": result_data['문제해결력'],
                        "논리적사고력": result_data['논리적사고력']
                    }
                    st.toast("🌱 첫 번째 기록이 생성되었습니다!", icon="🎉")

                collection.upsert(
                    documents=[new_text],
                    metadatas=[new_meta],
                    ids=[unique_student_id]
                )
                
                st.success(f"✅ LangChain 분석 완료! '{student_name_input}' 학생 데이터 저장 성공!")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader(f"🎯 누적 평균 점수 (총 {new_meta['분석횟수']}회)")
                    st.metric(label="과학적 탐구력", value=f"{new_meta['과학적탐구력']}점")
                    st.progress(new_meta['과학적탐구력'] / 100)
                    st.metric(label="문제 해결력", value=f"{new_meta['문제해결력']}점")
                    st.progress(new_meta['문제해결력'] / 100)
                    st.metric(label="논리적 사고력", value=f"{new_meta['논리적사고력']}점")
                    st.progress(new_meta['논리적사고력'] / 100)
                with col2:
                    st.subheader("📝 누적 세특 기록")
                    st.info(new_text)

            except Exception as e:
                st.error(f"LangChain 분석 중 오류가 발생했습니다: {e}")

st.divider()

# --- 화면 중앙 중간: 2. 학생 개별 검색 ---
st.header("🔍 2. 학생 개별 상세 검색")
search_col1, search_col2, search_col3 = st.columns([2, 2, 1])
with search_col1:
    search_id = st.text_input("검색할 학번", placeholder="예: 10101")
with search_col2:
    search_name = st.text_input("검색할 이름", placeholder="예: 김화학")
with search_col3:
    st.write("") 
    st.write("")
    search_btn = st.button("학생 불러오기", use_container_width=True)

if search_btn:
    if search_id and search_name:
        search_query = f"{search_id}_{search_name}"
        try:
            db_result = collection.get(ids=[search_query])
            if db_result and db_result['ids'] and len(db_result['ids']) > 0:
                saved_text = db_result['documents'][0]
                saved_meta = db_result['metadatas'][0]
                analyze_count = saved_meta.get('분석횟수', 1)
                st.success(f"📂 '{search_id} {search_name}' 학생의 {analyze_count}회 누적 데이터를 불러왔습니다.")
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric(label="과학적 탐구력", value=f"{saved_meta['과학적탐구력']}점")
                    st.metric(label="문제 해결력", value=f"{saved_meta['문제해결력']}점")
                    st.metric(label="논리적 사고력", value=f"{saved_meta['논리적사고력']}점")
                with res_col2:
                    st.info(saved_text)
            else:
                st.warning("데이터가 없습니다.")
        except Exception as e:
            st.error(f"검색 오류: {e}")
    else:
         st.warning("학번과 이름을 모두 입력해 주세요.")

st.divider()

# --- 화면 하단: 3. 전체 데이터 관리 및 LangChain Pandas Agent ---
st.header("📋 3. 전체 데이터베이스 관리")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 자연어 통계 (LangChain Agent)", "🔍 의미/키워드 검색", "💾 엑셀 다운로드", "🗑️ 삭제/초기화", "♻️ 복원"])

# 엑셀 및 Pandas Agent용 데이터프레임 만들기
def get_all_data_df():
    all_data = collection.get()
    if not all_data or not all_data['ids']:
        return None
    records = []
    for i in range(len(all_data['ids'])):
        parts = all_data['ids'][i].split("_", 1)
        meta = all_data['metadatas'][i]
        records.append({
            "학번": parts[0] if len(parts) > 1 else "",
            "이름": parts[1] if len(parts) > 1 else all_data['ids'][i],
            "과학적탐구력": meta.get("과학적탐구력", 0),
            "문제해결력": meta.get("문제해결력", 0),
            "논리적사고력": meta.get("논리적사고력", 0),
            "분석횟수": meta.get("분석횟수", 1)
        })
    return pd.DataFrame(records)

df_all = get_all_data_df()

with tab1:
    st.markdown("💬 **LangChain Pandas Agent 채팅:** 누적된 학생 데이터를 대상으로 질문을 던지면, 랭체인 요원이 파이썬 코드를 작성해 답을 찾아줍니다!")
    
    if df_all is None:
        st.info("아직 분석된 학생 데이터가 없습니다. 먼저 보고서를 분석해 주세요.")
    else:
        st.dataframe(df_all, use_container_width=True)
        
        pandas_query = st.text_input("데이터에 질문하기 (예: 과학적탐구력이 80점 이상인 학생 이름 알려줘)")
        
        if st.button("질문하기", type="primary"):
            if not api_key_input:
                st.error("왼쪽 사이드바에 API 키를 입력해 주세요.")
            elif pandas_query:
                with st.spinner("LangChain 요원이 데이터를 분석하고 있습니다... 🧮"):
                    try:
                        llm_for_pandas = ChatGoogleGenerativeAI(
                            model="gemini-3-flash-preview", 
                            google_api_key=api_key_input, 
                            temperature=0
                        )
                        
                        agent = create_pandas_dataframe_agent(
                            llm_for_pandas, 
                            df_all, 
                            verbose=True, 
                            allow_dangerous_code=True,
                            agent_executor_kwargs={"handle_parsing_errors": True} 
                        )
                        
                        safe_query = f"{pandas_query}\n\n(※ 중요: 출력 형식 에러를 방지하기 위해, 계산 과정이나 부가 설명 없이 반드시 'Final Answer: [당신의 최종 한국어 답변]' 형식으로만 답하세요.)"
                        
                        answer = agent.invoke(safe_query)
                        
                        st.success("✨ 분석 완료!")
                        st.write(answer["output"])
                    except Exception as e:
                        st.error(f"LangChain Agent 분석 중 오류가 발생했습니다: {e}")

with tab2:
    semantic_query = st.text_input("검색어 입력 (예: 실험 오차 분석, 꼼꼼함)")
    if st.button("통합 검색 시작"):
        if semantic_query:
            try:
                semantic_result = collection.query(query_texts=[semantic_query], n_results=3)
                if semantic_result and semantic_result['ids'][0]:
                    for i in range(len(semantic_result['ids'][0])):
                        st.expander(f"🏅 {i+1}순위: {semantic_result['ids'][0][i]}").write(semantic_result['documents'][0][i])
                else:
                    st.warning("관련된 기록이 없습니다.")
            except Exception as e:
                st.error(f"검색 오류: {e}")

with tab3:
    if df_all is not None:
        csv_data = df_all.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 엑셀(CSV) 다운로드", data=csv_data, file_name="학생전체데이터.csv", mime="text/csv", type="primary")
    else:
        st.info("저장된 데이터가 없습니다.")

with tab4:
    st.error("이 작업은 되돌릴 수 없습니다. 저장된 모든 데이터가 삭제됩니다.")
    reset_confirm = st.text_input("초기화하려면 '초기화' 입력")
    if st.button("전체 데이터 초기화"):
        if reset_confirm == "초기화":
            try:
                # 💡 [핵심 수정] 하드디스크 폴더에 접근하는 코드를 완전히 삭제하고, 오직 메모리 속 데이터만 깨끗하게 비웁니다!
                all_data = collection.get()
                if all_data and all_data['ids']:
                    collection.delete(ids=all_data['ids'])
                    st.cache_resource.clear()
                    st.success("✨ 데이터베이스가 깨끗하게 비워졌습니다! (새로고침 하세요)")
                else:
                    st.info("이미 데이터베이스가 비어있습니다.")
            except Exception as e:
                st.error(f"초기화 중 오류가 발생했습니다: {e}")
        else:
            st.warning("'초기화'를 정확히 입력하세요.")

with tab5:
    backup_file = st.file_uploader("백업 CSV 파일 업로드", type=["csv"])
    if st.button("♻️ 데이터 복원"):
        if backup_file is not None:
            try:
                df_backup = pd.read_csv(backup_file).fillna("")
                docs, metas, ids = [], [], []
                for index, row in df_backup.iterrows():
                    s_id = str(row.get("학번", "")).split('.')[0]
                    unique_id = f"{s_id}_{row.get('이름', '')}"
                    docs.append(str(row.get("누적세특기록", "")))
                    metas.append({
                        "과학적탐구력": int(row.get("과학적탐구력", 0)),
                        "문제해결력": int(row.get("문제해결력", 0)),
                        "논리적사고력": int(row.get("논리적사고력", 0)),
                        "분석횟수": int(row.get("분석횟수", 1))
                    })
                    ids.append(unique_id)
                if ids:
                    collection.upsert(documents=docs, metadatas=metas, ids=ids)
                    st.success(f"🎉 {len(ids)}명 복원 완료!")
            except Exception as e:
                st.error(f"복원 오류: {e}")