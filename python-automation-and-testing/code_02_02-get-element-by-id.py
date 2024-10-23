from selenium import webdriver

driver = webdriver.Firefox()
driver.get("http://localhost:8000/html_code_02.html")

login_form = driver.find_element("id", "loginForm")
print("My login form element is:")
print(login_form)
driver.close()
