## chatbot_ai
 
# Issues to look for

 1 - Make sure you install all the required modules to be able to run the application
 2 - To access the function inside the sql agent, covert the .ipynb to .py file, I already did that for you
 3 - I was able to run the flask using ```flask --app app run``` command, after installing flask module dependency


 # Some tips

 - Create python virtual env to install all deps 
  ``` Python3 -m venv your_env_name ``` your_env_name is your virtual environment name, just name it mybotenv or anything
 - Now you need to activate your env 
 ```source mybotenv/bin/activate```
 - Now when you have env activate, you will be able to see the env name on the left-most end of your terminal 
 - To install in module use pip, its package manager for python libraries like npm for JavaScript libraries
  ```pip install flask ```  you may use pip to install any kind of library
 - To convert ipynb to .py file use 
 - ```jupyter nbconvert --to script your_notebook.ipynb``` make sure you have ```pip install nbconvert``` installed