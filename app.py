import streamlit as st
import pandas as pd
import PyPDF2
import google.generativeai as genai
import json
import chromadb 
import os 
import shutil

# --- 데이터베이스 초기화 함수 ---
@st.cache_resource
def get_db_collection():
    client = chromadb.PersistentClient(path="./writable_db")
    collection = client.get_or_create_collection(name="chemistry_reports")
    return collection

collection = get_db_collection()

# 화면 기본 설정
st.set_page_config(page_title="생기부 작성 큐레이터", page_icon="🧪", layout="wide")

st.title("🧪 멀티모달 RAG 기반 학생 역량 큐레이터")
st.markdown("학생의 탐구 보고서(PDF)를 분석하여 핵심 역량을 추출하고 세특 데이터를 차곡차곡 누적합니다.")
st.divider()

# --- 화면 왼쪽 사이드바: 설정 및 입력 ---
with st.sidebar:
    st.header("⚙️ 기본 설정")
    
    API_KEY_FILE = "api_key.txt"
    saved_api_key = ""
    
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f:
            saved_api_key = f.read().strip()
            
    api_key_input = st.text_input("Gemini API 키 입력", type="password", value=saved_api_key, placeholder="API 키를 붙여넣으세요")
    
    if st.button("🔑 API 키 저장"):
        with open(API_KEY_FILE, "w") as f:
            f.write(api_key_input)
        st.success("API 키가 안전하게 저장되었습니다!")
    
    st.divider()
    
    st.header("📂 1. 새 보고서 분석")
    col_id, col_name = st.columns(2)
    with col_id:
        student_id_input = st.text_input("학번", placeholder="예: 10101")
    with col_name:
        student_name_input = st.text_input("이름", placeholder="예: 김화학")
        
    uploaded_file = st.file_uploader("PDF 보고서 업로드", type=["pdf"])
    
    analyze_btn = st.button("🚀 AI 분석 시작", type="primary", use_container_width=True)

# --- 함수: PDF에서 텍스트 읽어오기 ---
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- 화면 중앙 상단: 1. AI 분석 및 DB 저장 ---
st.header("📊 1. AI 분석 및 DB 누적 저장")

if analyze_btn:
    if not api_key_input:
        st.error("앗! 왼쪽 설정에서 Gemini API 키를 먼저 입력해 주세요.")
    elif not student_id_input or not student_name_input or not uploaded_file:
        st.warning("학번, 이름, 그리고 PDF 파일을 모두 확인해 주세요.")
    else:
        unique_student_id = f"{student_id_input}_{student_name_input}"
        
        with st.spinner(f"'{student_name_input}' 학생의 보고서를 AI가 분석 중입니다... 🤖"):
            try:
                document_text = extract_text_from_pdf(uploaded_file)
                
                genai.configure(api_key=api_key_input)
                model = genai.GenerativeModel('gemini-3-flash-preview') 
                
                # 💡 [수정 필요] 세부 역량 평가 기준을 선생님의 수업에 맞게 변경할 수 있습니다.
                prompt = f"""
                너는 통찰력 있는 고등학교 화학 교사야. 아래 학생이 제출한 화학 탐구 보고서를 읽어줘.
                이 보고서 내용을 바탕으로 다음 3가지 핵심 역량을 1점에서 100점 사이로 평가하고, 
                학교생활기록부 세부능력 및 특기사항(세특)에 들어갈 150자 이내의 아주 짧은 요약 문장을 명사형 종결어미(~함, ~임)로 작성해 줘.

                반드시 아래의 JSON(데이터) 형식으로만 대답해. 다른 설명은 덧붙이지 마.
                {{
                    "과학적탐구력": 85,
                    "문제해결력": 90,
                    "논리적사고력": 80,
                    "세특초안": "[보고서주제] 기체의 용해도와 온도의 상관관계를 파악하는 실험을 설계함."
                }}

                [학생 보고서 내용]
                {document_text}
                """
                
                response = model.generate_content(prompt)
                cleaned_response = response.text.replace('```json', '').replace('```', '').strip()
                result_data = json.loads(cleaned_response)
                
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
                
                st.success(f"✅ '{student_id_input} {student_name_input}' 학생의 데이터베이스 저장/누적 성공!")
                
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
                st.error(f"분석 중 오류가 발생했습니다: {e}")

st.divider()

# --- 화면 중앙 중간: 2. 학생 개별 검색 ---
st.header("🔍 2. 학생 개별 상세 검색")
st.markdown("특정 학생의 학번과 이름을 검색하여 상세 기록을 확인합니다.")

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
                st.warning(f"'{search_id} {search_name}' 학생의 데이터가 없습니다.")
        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {e}")
    else:
         st.warning("학번과 이름을 모두 입력해 주세요.")

st.divider()

# --- 화면 하단: 3. 전체 데이터 관리 및 삭제 ---
st.header("📋 3. 전체 데이터베이스 관리")
# 💡 [핵심 추가] 4번째 탭 "엑셀 백업본으로 복원" 기능이 추가되었습니다!
tab1, tab2, tab3, tab4 = st.tabs(["🔍 의미/키워드 통합 검색", "💾 엑셀(CSV) 전체 다운로드", "🗑️ 데이터 삭제/초기화", "♻️ 엑셀 백업본으로 복원"])

with tab1:
    st.markdown("단어나 문장을 입력하면, 그 의미와 가장 비슷한 세특 기록을 가진 학생을 순서대로 찾아줍니다.")
    semantic_query = st.text_input("검색어 입력 (예: 실험 오차 분석, 꼼꼼함, 화학 반응식 등)")
    
    if st.button("통합 검색 시작"):
        if semantic_query:
            with st.spinner("데이터베이스에서 의미를 분석하여 찾는 중입니다..."):
                try:
                    semantic_result = collection.query(
                        query_texts=[semantic_query],
                        n_results=3
                    )
                    
                    if semantic_result and semantic_result['ids'][0]:
                        st.success(f"'{semantic_query}'와(과) 관련된 기록을 가진 학생을 찾았습니다!")
                        
                        for i in range(len(semantic_result['ids'][0])):
                            student_id_name = semantic_result['ids'][0][i]
                            student_doc = semantic_result['documents'][0][i]
                            distance = round(semantic_result['distances'][0][i], 2)
                            
                            with st.expander(f"🏅 {i+1}순위: {student_id_name} (유사도 지수: {distance})"):
                                st.write(student_doc)
                    else:
                        st.warning("관련된 기록을 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"검색 중 오류가 발생했습니다: {e}")
        else:
            st.warning("검색어를 입력해 주세요.")

with tab2:
    st.markdown("현재까지 데이터베이스에 저장된 모든 학생의 역량 점수와 세특 기록을 엑셀 파일로 다운로드합니다.")
    
    if st.button("전체 데이터 가져오기"):
        all_data = collection.get()
        
        if all_data and all_data['ids']:
            records = []
            for i in range(len(all_data['ids'])):
                student_id_name = all_data['ids'][i]
                
                parts = student_id_name.split("_", 1)
                s_id = parts[0] if len(parts) > 1 else ""
                s_name = parts[1] if len(parts) > 1 else student_id_name
                
                meta = all_data['metadatas'][i]
                doc = all_data['documents'][i]
                
                records.append({
                    "학번": s_id,
                    "이름": s_name,
                    "과학적탐구력": meta.get("과학적탐구력", 0),
                    "문제해결력": meta.get("문제해결력", 0),
                    "논리적사고력": meta.get("논리적사고력", 0),
                    "분석횟수": meta.get("분석횟수", 1),
                    "누적세특기록": doc
                })
            
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
            
            csv_data = df.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 엑셀(CSV) 파일 다운로드",
                data=csv_data,
                file_name="학생역량분석_전체데이터.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.info("아직 저장된 학생 데이터가 없습니다.")

with tab3:
    st.markdown("특정 학생의 데이터를 삭제하거나, 새로운 학기를 맞이하여 전체 데이터베이스를 깨끗하게 비웁니다.")
    
    st.subheader("👤 개별 학생 데이터 삭제")
    del_col1, del_col2, del_col3 = st.columns([2, 2, 1])
    with del_col1:
        del_id = st.text_input("삭제할 학번", placeholder="예: 10101", key="del_id")
    with del_col2:
        del_name = st.text_input("삭제할 이름", placeholder="예: 김화학", key="del_name")
    with del_col3:
        st.write("") 
        st.write("")
        if st.button("개별 데이터 삭제"):
            if del_id and del_name:
                del_query = f"{del_id}_{del_name}"
                try:
                    existing_check = collection.get(ids=[del_query])
                    if existing_check and existing_check['ids']:
                        collection.delete(ids=[del_query])
                        st.success(f"🗑️ '{del_id} {del_name}' 학생의 데이터가 완전히 삭제되었습니다.")
                    else:
                        st.warning("해당 학생의 데이터가 존재하지 않습니다. 학번과 이름을 다시 확인해 주세요.")
                except Exception as e:
                    st.error(f"삭제 중 오류가 발생했습니다: {e}")
            else:
                st.warning("학번과 이름을 모두 입력해 주세요.")
                
    st.divider()
    
    st.subheader("🚨 전체 데이터베이스 초기화 (위험)")
    st.error("이 작업은 되돌릴 수 없습니다. 저장된 모든 학생의 데이터가 영구적으로 삭제됩니다.")
    reset_confirm = st.text_input("정말로 초기화하려면 아래 빈칸에 '초기화'라고 정확히 입력하세요.")
    
    if st.button("전체 데이터 초기화 실행", type="primary"):
        if reset_confirm == "초기화":
            try:
                if os.path.exists("./writable_db"):
                    shutil.rmtree("./writable_db")
                st.cache_resource.clear()
                st.success("✨ 데이터베이스가 깨끗하게 초기화되었습니다. 웹 브라우저를 새로고침(F5) 하시면 빈 금고로 다시 시작합니다!")
            except Exception as e:
                st.error(f"초기화 중 오류가 발생했습니다: {e}")
        else:
            st.warning("'초기화'라는 단어를 정확히 입력해야만 작동합니다.")

# 💡 [핵심 추가] 엑셀 파일을 읽어서 데이터베이스 금고에 다시 차곡차곡 넣어주는 마법의 기능입니다.
with tab4:
    st.markdown("클라우드 서버가 초기화되었을 때, 이전에 다운로드해둔 엑셀(CSV) 파일을 업로드하여 데이터를 완벽하게 복원합니다.")
    
    backup_file = st.file_uploader("다운로드했던 엑셀(CSV) 백업 파일 업로드", type=["csv"])
    
    if st.button("♻️ 데이터베이스 복원 실행", type="primary"):
        if backup_file is not None:
            try:
                # 엑셀(CSV) 파일을 읽어옵니다. (빈칸은 에러가 나지 않게 빈 문자열로 채워줍니다)
                df_backup = pd.read_csv(backup_file).fillna("")
                
                # 원본 파일이 맞는지 기둥(열 이름)을 검사합니다.
                required_cols = ["학번", "이름", "과학적탐구력", "문제해결력", "논리적사고력", "분석횟수", "누적세특기록"]
                if not all(col in df_backup.columns for col in required_cols):
                    st.error("잘못된 파일 형식입니다. 이 시스템에서 다운로드했던 원본 파일을 올려주세요.")
                else:
                    docs = []
                    metas = []
                    ids = []
                    
                    # 엑셀의 줄(행)을 하나씩 읽으면서 복원할 데이터를 만듭니다.
                    for index, row in df_backup.iterrows():
                        # 학번이 숫자로 저장된 경우 소수점(.0)이 생길 수 있으므로 깔끔하게 처리합니다.
                        s_id = str(row["학번"]).split('.')[0] if str(row["학번"]) else ""
                        s_name = str(row["이름"])
                        unique_id = f"{s_id}_{s_name}"
                        
                        doc = str(row["누적세특기록"])
                        meta = {
                            "과학적탐구력": int(row["과학적탐구력"]) if row["과학적탐구력"] else 0,
                            "문제해결력": int(row["문제해결력"]) if row["문제해결력"] else 0,
                            "논리적사고력": int(row["논리적사고력"]) if row["논리적사고력"] else 0,
                            "분석횟수": int(row["분석횟수"]) if row["분석횟수"] else 1
                        }
                        
                        docs.append(doc)
                        metas.append(meta)
                        ids.append(unique_id)
                        
                    if ids:
                        # 복원용 데이터를 한꺼번에 금고(DB)에 집어넣습니다!
                        collection.upsert(
                            documents=docs,
                            metadatas=metas,
                            ids=ids
                        )
                        st.success(f"🎉 총 {len(ids)}명의 학생 데이터가 성공적으로 복원되었습니다! 이제 다시 검색이 가능합니다.")
                    else:
                        st.warning("파일 안에 복원할 데이터가 없습니다.")
            except Exception as e:
                st.error(f"복원 중 오류가 발생했습니다: {e}")
        else:
            st.warning("먼저 CSV 파일을 업로드해 주세요.")