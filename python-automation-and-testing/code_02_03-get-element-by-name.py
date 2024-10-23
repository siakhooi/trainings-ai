from selenium import webdriver

driver = webdriver.Firefox()
driver.get("http://localhost:8000/html_code_02.html")
username = driver.find_element("name", "username")
print("My input element is:")
print(username)
driver.close()
