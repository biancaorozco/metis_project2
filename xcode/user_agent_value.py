from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

import chromedriver_binary



def get_user_agent():
    driver = webdriver.Chrome()

    #Store agent in a variable and print the value
    agent = driver.execute_script("return navigator.userAgent")
    return agent


