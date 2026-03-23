import streamlit as st
import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
import hashlib  # 【新增】用于密码哈希

# 导入自定义模块
from src.document_processor.processor import DocumentProcessor
from src.models.model_factory import ModelFactory
from src.vector_store.vector_store import VectorStore
from src.utils.helpers import get_available_models

# --- 【新增】安全与用户管理辅助函数 ---

def make_hashes(password):
    """生成密码的SHA256哈希值"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    """检查输入的密码是否匹配哈希值"""
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# 【重要】模拟数据库
# 在生产环境中，请将其替换为真实的数据库（如SQLite, MySQL, MongoDB）
# 格式: {"username": {"password": "hashed_password", "name": "用户昵称"}}
DB_USER_FILE = "users_db.csv"

def init_user_db():
    """初始化用户数据库（如果不存在）"""
    if not os.path.exists(DB_USER_FILE):
        df = pd.DataFrame(columns=["username", "password", "name"])
        df.to_csv(DB_USER_FILE, index=False)

def get_user_db():
    """读取用户数据库"""
    try:
        return pd.read_csv(DB_USER_FILE)
    except:
        return pd.DataFrame(columns=["username", "password", "name"])

def add_user_to_db(username, password, name):
    """注册新用户"""
    df = get_user_db()
    if username in df["username"].values:
        return False
    new_row = pd.DataFrame({
        "username": [username], 
        "password": [make_hashes(password)], 
        "name": [name]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DB_USER_FILE, index=False)
    return True

def authenticate_user(username, password):
    """验证用户登录"""
    df = get_user_db()
    if username in df["username"].values:
        stored_hash = df[df["username"] == username]["password"].values[0]
        if check_hashes(password, stored_hash):
            return True
    return False

def get_user_name(username):
    """获取用户昵称"""
    df = get_user_db()
    if username in df["username"].values:
        return df[df["username"] == username]["name"].values[0]
    return username

# --- 原有的辅助函数 ---

def generate_knowledge_profile(scores_df, model):
    """
    调用大语言模型，根据成绩DataFrame生成知识画像文本。
    """
    scores_text = scores_df.to_string(index=False)
    
    detailed_question = f"""
    请根据以下学生成绩数据，为学生生成一份全面、深刻且富有鼓励性的个人知识画像分析报告。

    **学生成绩数据：**
     **请按照以下结构生成报告，并严格遵守要求：** 
     {scores_text}
    1.  **整体表现概述**：用一句话总结学生的整体学习状况。
    2.  **优势科目分析**：识别出成绩最好的科目（例如85分以上），并分析这些科目可能反映出学生的哪些优点（如逻辑思维、记忆力、创造力等）。
    3.  **待提升科目分析**：识别出成绩相对薄弱的科目（例如70分以下），并分析可能存在的挑战或知识盲区。
    4.  **个性化学习建议**：基于以上分析，给出3-5条具体、可操作的学习建议。建议应涵盖：
        *   如何巩固优势科目。
        *   如何突破待提升科目。
        *   通用的学习方法和时间管理技巧。

    **要求：**
    *   语气要亲切、专业，像一位真正的导师。
    *   分析要深入，不要仅仅重复分数。
    *   建议要具体，避免空话套话。
    *   使用Markdown格式，用**加粗**突出重点。
    """

    system_prompt = "你是一位经验丰富的教育专家和学习分析师。你的任务是分析学生数据并提供专业、个性化的学习建议。"

    full_profile = ""
    for token in model.generate_stream_with_profile(
        question=detailed_question,
        profile_text=system_prompt
    ):
        full_profile += token
        
    return full_profile

def generate_enhanced_prompt(question: str, profile_text: str, relevant_docs: list) -> str:
    """
    构建一个结合了用户问题、知识画像和知识库内容的增强Prompt。
    """
    knowledge_content = ""
    if relevant_docs:
        doc_texts = []
        for doc in relevant_docs:
            if isinstance(doc, dict) and 'page_content' in doc:
                doc_texts.append(doc['page_content'])
            elif hasattr(doc, 'page_content'):
                doc_texts.append(doc.page_content)
            else:
                doc_texts.append("[无法解析的文档内容]")
        
        knowledge_content = "\n\n".join(doc_texts)
    else:
        knowledge_content = "抱歉，未在知识库中找到直接相关的内容。"

    enhanced_prompt = f"""
    你是一位专业的AI学习导师，现在需要回答一位学生提出的问题。你的回答必须同时基于以下三方面的信息：

    1.  **学生个人知识画像**：这能帮助你了解学生的长处和短板，进行个性化指导。
    2.  **通用教学知识库**：这为你提供了权威、全面的学习方法和知识点。
    3.  **学生当前问题**：这是你需要解决的核心任务。

    --- 学生知识画像 ---
    {profile_text}
    --- 画像结束 ---

    --- 相关知识库内容 ---
    {knowledge_content}
    --- 知识库内容结束 ---

    --- 学生当前问题 ---
    {question}
    --- 问题结束 ---

    **请根据以上所有信息，给出一个详细、个性化、可操作的回答。**
    - 你的回答应该直接引用或结合知识库中的内容。
    - 同时，你的建议必须紧密贴合学生的知识画像，例如，如果他在数学上是弱项，你的建议就应该更基础、更循序渐进。
    - 如果知识库中没有直接答案，请主要依据画像，结合你的专业知识进行回答。
    """
    return enhanced_prompt

# --- 页面配置 ---
st.set_page_config(
    page_title="教学诊断与学习指导助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 【新增】初始化用户数据库 ---
init_user_db()

# --- 【新增】登录/注册界面逻辑 ---
def show_login_page():
    # 使用HTML和CSS实现标题居中
    st.markdown("""
    <h1 style='text-align: center; color: #black;'>
        🧠 教学诊断与学习指导助手
    </h1>
    <hr style='border: 1px solid #eee; margin: 20px 0;'>
    """, unsafe_allow_html=True)
    # 使用列布局使界面居中
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab_login, tab_register = st.tabs(["用户登录", "新用户注册"])
        
        with tab_login:
            with st.form("login_form"):
                st.subheader("请登录")
                username = st.text_input("用户名", key="login_user")
                password = st.text_input("密码", type="password", key="login_pwd")
                login_btn = st.form_submit_button("登录", use_container_width=True)
                
                if login_btn:
                    if authenticate_user(username, password):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.session_state["user_display_name"] = get_user_name(username)
                        st.success("登录成功！正在跳转...")
                        st.rerun()  # 刷新页面进入主界面
                    else:
                        st.error("用户名或密码错误")

        with tab_register:
            with st.form("register_form"):
                st.subheader("创建账户")
                new_name = st.text_input("您的昵称", key="reg_name")
                new_user = st.text_input("设置用户名", key="reg_user")
                new_pwd = st.text_input("设置密码", type="password", key="reg_pwd")
                new_pwd_confirm = st.text_input("确认密码", type="password", key="reg_pwd_confirm")
                reg_btn = st.form_submit_button("注册", use_container_width=True)

                if reg_btn:
                    if not new_user or not new_pwd:
                        st.warning("用户名和密码不能为空")
                    elif new_pwd != new_pwd_confirm:
                        st.warning("两次输入的密码不一致")
                    else:
                        if add_user_to_db(new_user, new_pwd, new_name):
                            st.success("注册成功！请切换到登录页面登录。")
                        else:
                            st.error("用户名已存在，请更换一个")

# --- 初始化会话状态 ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_display_name" not in st.session_state:
    st.session_state.user_display_name = "访客"

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "current_document" not in st.session_state:
    st.session_state.current_document = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_processor" not in st.session_state:
    st.session_state.document_processor = DocumentProcessor()

if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "selected_model" not in st.session_state:
    try:
        available_models = get_available_models()
        st.session_state.selected_model = list(available_models.keys())[0]
    except Exception:
        st.session_state.selected_model = "gpt-3.5-turbo"

# --- 新增：成绩画像模块的会话状态 ---
if "student_scores_df" not in st.session_state:
    st.session_state.student_scores_df = None
if "knowledge_profile" not in st.session_state:
    st.session_state.knowledge_profile = None
if "profile_conversation_history" not in st.session_state:
    st.session_state.profile_conversation_history = []

if "profile_model" not in st.session_state:
    st.session_state.profile_model = None

# --- 核心控制流：判断显示登录页还是主应用 ---
if not st.session_state.logged_in:
    show_login_page()
else:
    # --- 以下是主应用界面 ---
    
    # --- 【修改】侧边栏显示用户信息 ---
    with st.sidebar:
        st.title("🧠 教学诊断与学习指导助手")
        st.subheader("教学诊断与学习指导助手")
        
        # 显示当前用户
        st.markdown(f"**👤 当前用户：** {st.session_state.user_display_name}")
        if st.button("退出登录"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            # 清空敏感数据
            st.session_state.knowledge_profile = None
            st.session_state.student_scores_df = None
            st.rerun()
        
        st.divider()

        # 模型选择
        try:
            available_models = get_available_models()
            selected_model = st.selectbox(
                "选择大语言模型",
                options=list(available_models.keys()),
                index=list(available_models.keys()).index(st.session_state.selected_model),
                help="选择用于回答问题的大语言模型"
            )
            st.session_state.selected_model = selected_model
        except Exception as e:
            st.error(f"无法加载模型列表: {e}")
            st.info("请检查 `.env` 文件或模型配置。")
        
        # 高级设置折叠面板
        with st.expander("高级设置"):
            top_k = st.slider(
                "检索文档数量", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.top_k,
                help="从文档或知识画像中检索的相关片段数量"
            )
            st.session_state.top_k = top_k
            
            temperature = st.slider(
                "温度", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.temperature, 
                step=0.1,
                help="控制回答的创造性"
            )
            st.session_state.temperature = temperature
        
        st.divider()
        st.caption("© 教学诊断与学习指导助手")

    # --- 加载教学知识库 ---
    if "knowledge_vector_store" not in st.session_state:
        st.session_state.knowledge_vector_store = None
        try:
            knowledge_base_filename = "knowledge_base.pdf"
            data_dir = "data"
            knowledge_base_path = os.path.join(data_dir, knowledge_base_filename)
            
            if os.path.exists(knowledge_base_path):
                with st.spinner("正在加载教学知识库..."):
                    document_processor = DocumentProcessor()
                    knowledge_chunks = document_processor.process_document(knowledge_base_path)
                    knowledge_vector_store = VectorStore()
                    knowledge_vector_store.add_documents(knowledge_chunks)
                    st.session_state.knowledge_vector_store = knowledge_vector_store
            else:
                st.sidebar.error(f"未找到知识库文件: `{knowledge_base_path}`")
        except Exception as e:
            st.sidebar.error(f"加载知识库时出错: {e}")
            st.session_state.knowledge_vector_store = None

    # --- 主界面 ---
    st.title("🧠 基于知识增强型大语言模型的智能学习助手")

    # 使用标签页组织不同功能
    tab1, tab2 = st.tabs(["📊 个人成绩画像", "📄 知识库上传"])

    with tab2:
        st.header("知识库管理")
        uploaded_file = st.file_uploader(
            "上传文档", 
            type=["pdf", "docx", "txt"], 
            key="doc_uploader",
            help="支持PDF、Word和TXT格式"
        )
        
        if uploaded_file and (uploaded_file.name != st.session_state.current_document):
            with st.spinner("正在处理文档..."):
                try:
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    document_processor = st.session_state.document_processor
                    document_chunks = document_processor.process_document(temp_path)
                    
                    vector_store = VectorStore()
                    vector_store.add_documents(document_chunks)
                    
                    st.session_state.current_document = uploaded_file.name
                    st.session_state.vector_store = vector_store
                    st.session_state.conversation_history = []
                    
                    st.success(f"文档 '{uploaded_file.name}' 已成功处理！共 {len(document_chunks)} 个片段。")
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                except Exception as e:
                    st.error(f"处理文档时出错: {str(e)}")
                    st.session_state.current_document = None
                    st.session_state.vector_store = None

        if st.session_state.current_document:
            st.info(f"✅ 当前文档: **{st.session_state.current_document}**")
        else:
            st.warning("⚠️ 请先上传文档以开始对话")

        for question, answer in st.session_state.conversation_history:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(question)
            with st.chat_message("assistant", avatar="🤖"):
                st.write(answer)

        if st.session_state.vector_store:
            question = st.chat_input("请输入您关于文档的问题...")
            if question:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(question)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("思考中..."):
                        try:
                            relevant_docs = st.session_state.vector_store.similarity_search(question, k=st.session_state.top_k)
                            model_info = available_models[st.session_state.selected_model]
                            model = ModelFactory.get_model(
                                model_type=model_info["type"],
                                model_name=model_info["name"],
                                temperature=st.session_state.temperature
                            )
                            answer_container = st.empty()
                            full_answer = ""
                            for token in model.generate_stream(question, relevant_docs):
                                full_answer += token
                                answer_container.markdown(full_answer + "▌")
                            answer_container.markdown(full_answer)
                            st.session_state.conversation_history.append((question, full_answer))
                        except Exception as e:
                            st.error(f"生成回答时出错: {str(e)}")
        else:
            st.info("请先上传文档以开始对话")


    with tab1:
        st.header("个人成绩画像与问答")
        
        st.subheader("1. 录入成绩数据")
        input_method = st.radio("选择输入方式", ["手动输入", "上传CSV文件"])

        scores_df = None
        if input_method == "手动输入":
            with st.form("manual_score_form"):
                st.write("请在下方输入您的科目和成绩：")
                num_subjects = st.number_input("科目数量", min_value=1, max_value=10, value=3)
                subjects = []
                scores = []
                for i in range(num_subjects):
                    col1, col2 = st.columns(2)
                    with col1:
                        subjects.append(st.text_input(f"科目 {i+1}", key=f"subject_{i}"))
                    with col2:
                        scores.append(st.number_input(f"成绩 {i+1}", min_value=0.0, max_value=100.0, key=f"score_{i}"))
                
                submitted = st.form_submit_button("生成知识画像")
                if submitted:
                    if all(subjects) and all(s is not None for s in scores):
                        scores_df = pd.DataFrame({"科目": subjects, "成绩": scores})
                        st.session_state.student_scores_df = scores_df
                    else:
                        st.error("请填写所有科目和成绩！")

        else: 
            uploaded_csv = st.file_uploader("上传成绩CSV文件", type="csv", help="CSV文件应包含'科目'和'成绩'两列")
            if uploaded_csv:
                try:
                    scores_df = pd.read_csv(uploaded_csv)
                    if '科目' not in scores_df.columns or '成绩' not in scores_df.columns:
                        st.error("CSV文件必须包含 '科目' 和 '成绩' 两列！")
                        scores_df = None
                    else:
                        st.success("成绩数据读取成功！")
                        st.session_state.student_scores_df = scores_df
                except Exception as e:
                    st.error(f"读取CSV文件失败: {e}")
                    scores_df = None

        if st.session_state.student_scores_df is not None:
            if st.session_state.profile_model is None:
                try:
                    model_info = available_models[st.session_state.selected_model]
                    st.session_state.profile_model = ModelFactory.get_model(
                        model_type=model_info["type"],
                        model_name=model_info["name"],
                        temperature=st.session_state.temperature
                    )
                except Exception as e:
                    st.error(f"无法初始化模型来生成画像: {e}")
                    st.session_state.profile_model = None

            if st.session_state.knowledge_profile is None and st.session_state.profile_model is not None:
                with st.spinner("AI正在分析您的成绩，生成个性化知识画像..."):
                    profile = generate_knowledge_profile(st.session_state.student_scores_df, st.session_state.profile_model)
                    st.session_state.knowledge_profile = profile
                
            if st.session_state.knowledge_profile:
                st.subheader("2. 您的AI知识画像")
                st.info(st.session_state.knowledge_profile)

            if st.session_state.profile_model is not None and st.session_state.knowledge_profile is not None:
                st.subheader("3. 基于画像提问 (结合知识库)")
                for q, a in st.session_state.profile_conversation_history:
                    with st.chat_message("user", avatar="🧑‍🎓"):
                        st.write(q)
                    with st.chat_message("assistant", avatar="🧠"):
                        st.write(a)

                profile_question = st.chat_input("请输入您关于学习的问题，例如：'我该如何提高数学？'", key="profile_chat_input")
                if profile_question:
                    with st.chat_message("user", avatar="🧑‍🎓"):
                        st.write(profile_question)
                    
                    with st.chat_message("assistant", avatar="🧠"):
                        with st.spinner("正在结合您的知识画像和知识库进行分析..."):
                            try:
                                if not st.session_state.knowledge_vector_store:
                                    st.warning("知识库未加载，将仅基于您的画像进行回答。")
                                    relevant_docs = []
                                else:
                                    relevant_docs = st.session_state.knowledge_vector_store.similarity_search(
                                        profile_question, 
                                        k=st.session_state.top_k
                                    )

                                enhanced_prompt = generate_enhanced_prompt(
                                    question=profile_question,
                                    profile_text=st.session_state.knowledge_profile,
                                    relevant_docs=relevant_docs
                                )

                                answer_container = st.empty()
                                full_answer = ""
                                
                                for token in st.session_state.profile_model.generate_stream(enhanced_prompt, context=[]):
                                    full_answer += token
                                    answer_container.markdown(full_answer + "▌")
                                
                                answer_container.markdown(full_answer)
                                
                                st.session_state.profile_conversation_history.append((profile_question, full_answer))
                            except Exception as e:
                                st.error(f"生成回答时出错: {str(e)}")
            elif st.session_state.profile_model is None:
                st.error("由于模型初始化失败，无法进行知识画像问答。请检查配置或刷新页面重试。")

        else:
            st.info("请先输入您的成绩数据以生成知识画像。")
