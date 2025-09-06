"""
Chat interface with LangChain text-to-SQL for querying financial data.
Allows users to ask natural language questions about their data.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from sqlalchemy import create_engine, text, inspect
from langchain_openai import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLQueryParser(BaseOutputParser):
    """Custom parser for SQL query results."""
    
    def parse(self, text: str) -> str:
        """Parse the LLM output and extract meaningful response."""
        # Remove SQL query from response if present
        lines = text.strip().split('\n')
        
        # Look for the actual answer after SQL execution
        answer_started = False
        answer_lines = []
        
        for line in lines:
            if 'Answer:' in line or answer_started:
                answer_started = True
                if 'Answer:' in line:
                    answer_lines.append(line.replace('Answer:', '').strip())
                else:
                    answer_lines.append(line.strip())
        
        if answer_lines:
            return ' '.join(answer_lines)
        
        # Fallback: return the last non-empty line
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith('SELECT'):
                return line.strip()
        
        return text.strip()


class FinancialChatInterface:
    """Chat interface for financial data queries using LangChain and text-to-SQL."""
    
    def __init__(self, db_path: str = "db/altsignals.db", openai_api_key: str = None):
        self.db_path = db_path
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.openai_client = OpenAIClient(api_key=self.openai_api_key)
        
        # Setup database connection
        self.setup_database()
        
        # Initialize LangChain components
        self.setup_langchain()
        
        logger.info("FinancialChatInterface initialized")
    
    def setup_database(self):
        """Setup database connection and inspect schema."""
        try:
            # Create SQLAlchemy engine
            db_url = f"sqlite:///{self.db_path}"
            self.engine = create_engine(db_url)
            
            # Create LangChain SQLDatabase
            self.sql_database = SQLDatabase(self.engine)
            
            # Get table information
            self.table_info = self.get_table_info()
            
            logger.info(f"Connected to database: {self.db_path}")
            logger.info(f"Available tables: {list(self.table_info.keys())}")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def setup_langchain(self):
        """Setup LangChain components."""
        try:
            # Create custom prompt template for financial queries
            template = """
            You are a financial data analyst AI assistant. You have access to a financial database with the following tables and schema:

            {table_info}

            Given a user question about financial data, write a SQL query to answer the question.
            Then provide a clear, human-readable answer based on the query results.

            Important guidelines:
            1. Only use tables and columns that exist in the schema above
            2. Be precise with column names and table names
            3. Use appropriate SQL functions for calculations (AVG, SUM, COUNT, etc.)
            4. Format numbers appropriately (e.g., currency, percentages)
            5. If the question cannot be answered with available data, explain what's missing
            6. Provide context and interpretation of the results

            Question: {input}

            SQL Query:
            """

            self.prompt_template = PromptTemplate(
                input_variables=["input", "table_info"],
                template=template
            )
            
            # Create SQL database chain
            self.sql_chain = SQLDatabaseChain.from_llm(
                llm=OpenAI(
                    api_key=self.openai_api_key,
                    base_url="https://api.openai.com/v1",
                    temperature=0,
                    model="gpt-3.5-turbo-instruct"
                ),
                db=self.sql_database,
                prompt=self.prompt_template,
                verbose=True,
                return_intermediate_steps=True
            )
            
        except Exception as e:
            logger.error(f"Error setting up LangChain: {e}")
            raise
    
    def get_table_info(self) -> Dict[str, Dict]:
        """Get detailed information about database tables."""
        table_info = {}
        
        try:
            inspector = inspect(self.engine)
            
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                table_info[table_name] = {
                    'columns': [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col['nullable']
                        }
                        for col in columns
                    ]
                }
                
                # Get sample data
                try:
                    with self.engine.connect() as conn:
                        sample_query = text(f"SELECT * FROM {table_name} LIMIT 3")
                        result = conn.execute(sample_query)
                        sample_data = result.fetchall()
                        table_info[table_name]['sample_data'] = [dict(row._mapping) for row in sample_data]
                except Exception as e:
                    logger.warning(f"Could not get sample data for {table_name}: {e}")
                    table_info[table_name]['sample_data'] = []
            
            return table_info
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {}
    
    def format_table_info_for_prompt(self) -> str:
        """Format table information for the prompt."""
        formatted_info = []
        
        for table_name, info in self.table_info.items():
            formatted_info.append(f"\nTable: {table_name}")
            formatted_info.append("Columns:")
            
            for col in info['columns']:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                formatted_info.append(f"  - {col['name']} ({col['type']}) {nullable}")
            
            # Add sample data if available
            if info['sample_data']:
                formatted_info.append("Sample data:")
                for i, row in enumerate(info['sample_data'][:2]):
                    formatted_info.append(f"  Row {i+1}: {row}")
        
        return '\n'.join(formatted_info)
    
    def query_database_direct(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query directly and return results."""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(sql_query, conn)
                return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a natural language question about the financial data."""
        try:
            # Clean the question to handle Unicode characters
            clean_question = question.encode('ascii', 'ignore').decode('ascii')
            if not clean_question.strip():
                clean_question = "What data do we have?"
            
            logger.info(f"Processing question: {clean_question}")
            
            # Format table info for the prompt
            table_info_str = self.format_table_info_for_prompt()
            
            # Use the SQL chain to process the question
            result = self.sql_chain({
                "query": clean_question,
                "table_info": table_info_str
            })
            
            # Extract components from result
            sql_query = ""
            answer = result.get('result', 'No answer generated')
            
            # Try to extract SQL query from intermediate steps
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    if isinstance(step, dict) and 'sql_cmd' in step:
                        sql_query = step['sql_cmd']
                        break
            
            # Get query results if SQL was executed
            query_results = None
            if sql_query:
                try:
                    query_results = self.query_database_direct(sql_query)
                except Exception as e:
                    logger.warning(f"Could not re-execute query: {e}")
            
            return {
                'question': question,
                'answer': answer,
                'sql_query': sql_query,
                'query_results': query_results,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'question': question,
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sql_query': "",
                'query_results': None,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions based on available data."""
        suggestions = [
            "What stocks do we have data for?",
            "Show me the latest prices for all stocks",
            "Which stock has the highest current price?",
            "What is the average PE ratio across all stocks?",
            "Show me stocks with PE ratio less than 20",
            "What are the recent trading signals?",
            "Which stocks have positive sentiment?",
            "Show me the most volatile stocks",
            "What is the total market cap of all stocks?",
            "Which sectors are represented in our data?",
            "Show me stocks with dividend yield above 2%",
            "What are the latest news articles?",
            "Which stocks have the highest volume?",
            "Show me backtest results",
            "What options data do we have?"
        ]
        
        return suggestions
    
    def explain_database_schema(self) -> str:
        """Provide an explanation of the database schema."""
        explanation = """
        **AltSignals Database Schema:**
        
        **stocks**: Main stock information table
        - Contains fundamental data like PE ratios, market cap, sector, etc.
        - Key fields: symbol, name, current_price, pe_ratio, market_cap
        
        **stock_prices**: Historical price data
        - Daily OHLCV data for each stock
        - Key fields: date, open_price, high_price, low_price, close_price, volume
        
        **signals**: Trading signals and recommendations
        - Generated signals with strength and confidence scores
        - Key fields: signal_type, strength, confidence, target_price
        
        **news**: News articles and sentiment analysis
        - News articles with sentiment scores
        - Key fields: title_en, sentiment_score, sentiment_label, published_at
        
        **backtest_results**: Strategy backtesting results
        - Performance metrics from strategy backtests
        - Key fields: strategy_name, total_return, sharpe_ratio, max_drawdown
        
        **options**: Options data and Greeks
        - Options pricing and risk metrics
        - Key fields: option_type, strike_price, delta, gamma, theta, vega
        
        You can ask questions about any of this data!
        """
        
        return explanation
    
    def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick statistics about the database."""
        stats = {}
        
        try:
            with self.engine.connect() as conn:
                # Count records in each table
                for table_name in self.table_info.keys():
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = result.scalar()
                        stats[f"{table_name}_count"] = count
                    except Exception as e:
                        logger.warning(f"Could not count records in {table_name}: {e}")
                        stats[f"{table_name}_count"] = 0
                
                # Get date range for stock prices
                try:
                    result = conn.execute(text("SELECT MIN(date), MAX(date) FROM stock_prices"))
                    min_date, max_date = result.fetchone()
                    stats['price_data_range'] = {
                        'start': str(min_date) if min_date else None,
                        'end': str(max_date) if max_date else None
                    }
                except Exception as e:
                    logger.warning(f"Could not get price data range: {e}")
                    stats['price_data_range'] = None
                
                # Get unique symbols
                try:
                    result = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM stocks"))
                    unique_symbols = result.scalar()
                    stats['unique_symbols'] = unique_symbols
                except Exception as e:
                    logger.warning(f"Could not count unique symbols: {e}")
                    stats['unique_symbols'] = 0
        
        except Exception as e:
            logger.error(f"Error getting quick stats: {e}")
        
        return stats


def create_chat_interface(db_path: str = "db/altsignals.db") -> FinancialChatInterface:
    """Factory function to create chat interface."""
    return FinancialChatInterface(db_path)


if __name__ == "__main__":
    # Test the chat interface
    try:
        print("Testing Financial Chat Interface...")
        
        chat = FinancialChatInterface()
        
        # Test database connection
        print("\nDatabase Schema:")
        print(chat.explain_database_schema())
        
        # Test quick stats
        print("\nQuick Stats:")
        stats = chat.get_quick_stats()
        for key, value in stats.items():
            print(f"- {key}: {value}")
        
        # Test sample questions
        sample_questions = [
            "What stocks do we have data for?",
            "Show me the current prices for all stocks",
            "Which stock has the highest PE ratio?"
        ]
        
        print("\nTesting sample questions:")
        for question in sample_questions:
            print(f"\nQ: {question}")
            try:
                response = chat.ask_question(question)
                print(f"A: {response['answer']}")
                if response['sql_query']:
                    print(f"SQL: {response['sql_query']}")
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nChat interface testing completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure the database exists and OPENAI_API_KEY is set.")

