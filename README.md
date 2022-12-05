# install the requirements

pip install -r requirements.txt

# or with virtualenv 

virtualenv venv

source venv/bin/activate

pip install -r requirements.txt

# run the app

uvicorn main:app --reload

# go to http://127.0.0.1:8000

# live api docs at:

https://anun-production.up.railway.app/docs
