# KekBootCamp
Kek



git clone https://github.com/yourusername/research-pro-mode.git
cd research-pro-mode

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


pip install streamlit langchain-core langchain-openai tavily-python requests beautifulsoup4 pydantic pandas python-dotenv

API_KEY=твой_ключ_от_cloud_ru
TAVILY_API_KEY=твой_ключ_от_tavily
DEBUG_MODE=true  # Опционально, для отладки  

streamlit run app.py
