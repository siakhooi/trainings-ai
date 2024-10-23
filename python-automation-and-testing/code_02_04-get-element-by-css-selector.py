from selenium import webdriver

driver = webdriver.Firefox()
driver.get("http://localhost:8000/html_code_02.html")
content = driver.find_element("css selector", ".content")
print("My class element is:")
print(content)
driver.close()
