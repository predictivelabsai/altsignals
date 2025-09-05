"""
Chat interface page for natural language queries about financial data.
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from chat_interface import FinancialChatInterface

st.set_page_config(page_title="Chat Interface", page_icon="💬", layout="wide")

def initialize_chat_interface():
    """Initialize the chat interface."""
    if 'chat_interface' not in st.session_state:
        try:
            st.session_state.chat_interface = FinancialChatInterface()
            st.session_state.chat_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize chat interface: {e}")
            st.session_state.chat_initialized = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_chat_message(message, is_user=True):
    """Display a chat message."""
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)

def display_query_results(response):
    """Display query results in a structured way."""
    if response.get('query_results') is not None and not response['query_results'].empty:
        st.subheader("Query Results")
        st.dataframe(response['query_results'], use_container_width=True)
    
    if response.get('sql_query'):
        with st.expander("View SQL Query"):
            st.code(response['sql_query'], language='sql')

def main():
    st.title("💬 Financial Data Chat")
    st.markdown("Ask questions about your financial data in natural language!")
    
    # Initialize chat interface
    initialize_chat_interface()
    
    if not st.session_state.get('chat_initialized', False):
        st.error("Chat interface is not available. Please check your database connection and API keys.")
        return
    
    # Sidebar with information and suggestions
    with st.sidebar:
        st.header("💡 How to Use")
        st.markdown("""
        Ask questions about your financial data in plain English:
        
        **Examples:**
        - "What stocks do we have data for?"
        - "Show me the latest prices"
        - "Which stock has the highest PE ratio?"
        - "What are the recent trading signals?"
        - "Show me stocks with positive sentiment"
        """)
        
        st.header("📊 Database Info")
        try:
            chat = st.session_state.chat_interface
            stats = chat.get_quick_stats()
            
            st.metric("Total Stocks", stats.get('stocks_count', 0))
            st.metric("Price Records", stats.get('stock_prices_count', 0))
            st.metric("News Articles", stats.get('news_count', 0))
            st.metric("Trading Signals", stats.get('signals_count', 0))
            
            if stats.get('price_data_range'):
                date_range = stats['price_data_range']
                if date_range['start'] and date_range['end']:
                    st.write(f"**Data Range:** {date_range['start']} to {date_range['end']}")
        
        except Exception as e:
            st.error(f"Error loading database stats: {e}")
        
        st.header("🔍 Suggested Questions")
        suggested_questions = [
            "What stocks do we have?",
            "Show latest prices",
            "Highest PE ratio stock?",
            "Recent trading signals?",
            "Stocks with positive sentiment?",
            "Most volatile stocks?",
            "Average market cap?",
            "Latest news articles?"
        ]
        
        for question in suggested_questions:
            if st.button(question, key=f"suggest_{question}"):
                st.session_state.user_input = question
                st.rerun()
    
    # Main chat interface
    st.header("Chat History")
    
    # Display chat history
    for i, (question, response) in enumerate(st.session_state.chat_history):
        display_chat_message(question, is_user=True)
        display_chat_message(response['answer'], is_user=False)
        
        if response.get('query_results') is not None and not response['query_results'].empty:
            with st.expander(f"Results for: {question[:50]}..."):
                st.dataframe(response['query_results'], use_container_width=True)
                
                if response.get('sql_query'):
                    st.code(response['sql_query'], language='sql')
        
        st.divider()
    
    # Chat input
    user_input = st.chat_input("Ask a question about your financial data...")
    
    # Handle suggested question from sidebar
    if 'user_input' in st.session_state:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    
    if user_input:
        # Display user message
        display_chat_message(user_input, is_user=True)
        
        # Process the question
        with st.spinner("Thinking..."):
            try:
                chat = st.session_state.chat_interface
                response = chat.ask_question(user_input)
                
                # Display assistant response
                display_chat_message(response['answer'], is_user=False)
                
                # Display query results if available
                display_query_results(response)
                
                # Add to chat history
                st.session_state.chat_history.append((user_input, response))
                
                # Limit chat history to last 10 exchanges
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
                
            except Exception as e:
                st.error(f"Error processing your question: {e}")
                display_chat_message(f"Sorry, I encountered an error: {str(e)}", is_user=False)
        
        st.rerun()
    
    # Database schema information
    with st.expander("📋 Database Schema Information"):
        try:
            chat = st.session_state.chat_interface
            schema_info = chat.explain_database_schema()
            st.markdown(schema_info)
        except Exception as e:
            st.error(f"Error loading schema information: {e}")
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**💬 Financial Chat Interface** | Powered by LangChain & OpenAI GPT")
    st.markdown("*Ask questions about stocks, prices, signals, news, and more!*")

if __name__ == "__main__":
    main()

