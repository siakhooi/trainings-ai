from selenium import webdriver
import time
from selenium.webdriver.support.ui import Select

driver = webdriver.Firefox()
driver.get("http://localhost:8000/html_code_03_02.html")

select = Select(driver.find_element("name", "numReturnSelect"))
select.select_by_index(4)
time.sleep(2)
select.select_by_visible_text("200")
time.sleep(2)
select.select_by_value("250")
time.sleep(2)

options = select.options
print(options)

submit_button = driver.find_element('name', "continue")
submit_button.submit()
time.sleep(2)

driver.close()
