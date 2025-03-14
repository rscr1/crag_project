import asyncio
import streamlit as st
import sqlite3
from datetime import datetime
from web_search_rag import app
# from plat_rag import app

def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  time TIMESTAMP,
                  title TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chat_id INTEGER,
                  role TEXT,
                  content TEXT,
                  timestamp TIMESTAMP,
                  FOREIGN KEY(chat_id) REFERENCES chats(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS context
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  type TEXT,
                  message_id INTEGER,
                  title TEXT,
                  source TEXT,
                  content TEXT,
                  timestamp TIMESTAMP,
                  FOREIGN KEY(message_id) REFERENCES messages(id))''')
    conn.commit()
    return conn

def delete_empty_chat(conn, chat_id):
    c = conn.cursor()
    message_count = c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)).fetchone()[0]
    if message_count == 0:
        c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
        return True
    return False

async def run_interface(app):
    conn = init_db()
    c = conn.cursor()
    
    with st.sidebar:
        chats = c.execute('''
            SELECT id, title FROM chats 
            WHERE id IN (
                SELECT chat_id FROM messages GROUP BY chat_id HAVING COUNT(*) > 1
            )
            ORDER BY time DESC
        ''').fetchall()
  
        if st.button("➕ New Chat"):
            if 'current_chat' in st.session_state:
                delete_empty_chat(conn, st.session_state.current_chat)
            now = datetime.now()
            c.execute("INSERT INTO chats (time, title) VALUES (?, ?)", (now, "New Chat"))
            conn.commit()
            st.session_state.current_chat = c.lastrowid
            st.rerun()

        st.markdown("<h2 style='text-align: center;'>Chat History</h2>", unsafe_allow_html=True)
        if chats:
            for chat_id, title in chats:
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(f"💬 {title}", key=f"chat_{chat_id}", use_container_width=True):
                        if 'current_chat' in st.session_state and st.session_state.current_chat != chat_id:
                            delete_empty_chat(conn, st.session_state.current_chat)
                        st.session_state.current_chat = chat_id
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}"):
                        c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
                        c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                        conn.commit()
                        if 'current_chat' in st.session_state and st.session_state.current_chat == chat_id:
                            remaining_chats = c.execute("SELECT id FROM chats ORDER BY time DESC").fetchall()
                            st.session_state.current_chat = remaining_chats[0][0] if remaining_chats else None
                        st.rerun()        

    st.title("Assistant")
    
    if 'current_chat' not in st.session_state or not st.session_state.current_chat:
        now = datetime.now()
        c.execute("INSERT INTO chats (time, title) VALUES (?, ?)", (now, "New Chat"))
        conn.commit()
        st.session_state.current_chat = c.lastrowid

    messages = c.execute("""
        SELECT id, role, content 
        FROM messages 
        WHERE chat_id = ? 
        ORDER BY timestamp
    """, (st.session_state.current_chat,)).fetchall()
    
    for message_id, role, content in messages:
        if role == 'user':
            st.markdown(f"""
            <div style="text-align: left; margin: 10px; padding: 10px; background-color: #0d6efd; color: white; border-radius: 10px; display: inline-block; max-width: 80%; float: right;">
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(content)
            response = c.execute("""
                SELECT type, title, source, content
                FROM context 
                WHERE message_id = ?
                ORDER BY timestamp
            """, (message_id,)).fetchall()
            
            if response:
                with st.expander("Retrieval Context ▼", expanded=False):
                    for i, (ctype, title, source, ccontent) in enumerate(response, 1):
                        if ctype == 'doc':
                            if i == 1:
                                st.write("**📄 Relevant Documents:**")
                            st.markdown(f"""
                            <div style="
                                padding: 10px; 
                                background: #000000;
                                border-radius: 5px; 
                                margin: 5px 0;
                                font-size: 0.9em;
                                color: #ffffff;
                            ">
                            <b>Document {title}:</b><br>
                            {ccontent}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if i == 1:
                                st.write("**🌐 Web Results:**")
                            st.markdown(f"""
                            <div style="
                                padding: 10px;
                                background: #000000;
                                border-radius: 5px;
                                margin: 5px 0;
                                font-size: 0.9em;
                                color: #ffffff;
                            ">
                            <b>Result {i}: {title}</b><br>
                            {ccontent}
                            </div>
                            """, unsafe_allow_html=True)
                            st.write("🔗 Source:", source)

    prompt = st.chat_input("Ask anything")
    
    if prompt:
        st.markdown(f"""
            <div style="text-align: right; margin: 10px; padding: 10px; background-color: #0d6efd; color: white; border-radius: 10px; display: inline-block; max-width: 80%; float: right;">
                {prompt}
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("Analyzing..."):
            c.execute("""
                INSERT INTO messages (chat_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (st.session_state.current_chat, 'user', prompt, datetime.now()))
            conn.commit()
            
            # Обновляем заголовок чата, если это первое сообщение
            if c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (st.session_state.current_chat,)).fetchone()[0] == 1:
                shortened_prompt = (prompt[:20] + '...') if len(prompt) > 20 else prompt
                title = f"Chat: {shortened_prompt}"
                c.execute("UPDATE chats SET title = ? WHERE id = ?", (title, st.session_state.current_chat))
                conn.commit()
            
            initial_state = {"question": prompt}
            response = await app.ainvoke(initial_state)
            answer = response['generation'].response
            
            c.execute("""
                INSERT INTO messages (chat_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (st.session_state.current_chat, 'assistant', answer, datetime.now()))
            assistant_message_id = c.lastrowid
            conn.commit()

            st.markdown(answer)
            
            with st.expander("Retrieval Context ▼", expanded=False):
                if response.get('documents') and len(response['documents']) != 0:
                    st.write("**📄 Relevant Documents:**")
                    for i, doc in enumerate(response['documents'], 1):
                        st.markdown(f"""
                        <div style="
                            padding: 10px; 
                            background: #000000;
                            border-radius: 5px; 
                            margin: 5px 0;
                            font-size: 0.9em;
                            color: #ffffff;
                        ">
                        <b>Document {i}:</b><br>
                        {doc.text}
                        </div>
                        """, unsafe_allow_html=True)
                        c.execute("""
                            INSERT INTO context (message_id, type, title, source, content, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (assistant_message_id, 'doc', i, None, doc.text, datetime.now()))
                        conn.commit()
                
                if 'web_results' in response:
                    st.write("**🌐 Web Results:**")
                    for i, page in enumerate(response['web_results'], 1):
                        st.markdown(f"""
                        <div style="
                            padding: 10px;
                            background: #000000;
                            border-radius: 5px;
                            margin: 5px 0;
                            font-size: 0.9em;
                            color: #ffffff;
                        ">
                        <b>Result {i}: {page.node.metadata.get('title', 'No Title')}</b><br>
                        {page.node.metadata.get('content', 'No content')}
                        </div>
                        """, unsafe_allow_html=True)
                        st.write("🔗 Source:", page.node.metadata.get('url', 'No URL available'))
                        c.execute("""
                            INSERT INTO context (message_id, type, title, source, content, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (assistant_message_id, 'web', page.node.metadata.get('title', 'No Title'), page.node.metadata.get('url', 'No URL available'), page.node.metadata.get('content', 'No content'), datetime.now()))
                        conn.commit()
            st.rerun()

if __name__ == "__main__":
    asyncio.run(run_interface(app))
