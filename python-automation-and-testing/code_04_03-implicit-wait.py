from selenium import webdriver

driver = webdriver.Firefox()

driver.implicitly_wait(10)

driver.get("http://www.python.org")
myDynamicElement = driver.find_element("id", "start-shell")

driver.close()
