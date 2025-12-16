# LMKRChatbot
A chatbot that answers questions about the company based on its RAG model.

To run:
1. First make a virtual environment using python -m venv venv

2. For windows set execution policy: -ExecutionPolicy RemoteSigned -Scope CurrentUser

3. Then activate the environment::
For windows:> .\venv\Scripts\activate        
For Linux:  source venv/bin/activate

4. Install requirements: pip install -r requirement.txt

5. Run the backend: uvicorn main:api --reload

6. Run the frontend: streamlit run frontend.py      