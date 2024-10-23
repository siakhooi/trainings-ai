from selenium import webdriver
import time

driver = webdriver.Edge()
driver.get("https://selenium.dev")
time.sleep(2)

driver.close()
