from selenium import webdriver

driver = webdriver.Firefox()

driver.get("http://localhost:8000/html_code_02.html")
# absolute xpath
login_form_absolute = driver.find_element("xpath", "/html/body/form[1]")
# relative xpath
login_form_relative = driver.find_element("xpath", "//form[1]")
# relative xpath, with id
login_form_id = driver.find_element("xpath", "//form[@id='loginForm']")
print("My login form is:")
print(login_form_absolute)
print(login_form_relative)
print(login_form_id)
driver.close()
