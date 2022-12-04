from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os


more_button = 'button2 button2_size_l button2_theme_action button2_type_link button2_view_classic more__button i-bem button2_js_inited'


def make_category(browser_: webdriver, category: str):
    url = f'https://yandex.ru/images/search?text={category}'
    browser_.get(url)
    time.sleep(1)

    for i in range(700):
        browser_.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        if i % 30 == 0:
            time.sleep(1)
        try:
            browser_.find_element(
                by=By.CLASS_NAME,
                value='more more_direction_next i-bem more_js_inited more_last_yes'
            ).find_element(
                by=By.CLASS_NAME,
                value=more_button,
            ).click()
        except:
            pass

    a_tags = browser_.find_elements(
        by=By.CLASS_NAME,
        value='serp-item__link',
    )
    print(f'For category {category} is found {len(a_tags)} images')
    links = [a.find_element(By.TAG_NAME, 'img').get_attribute('src') + '\n'
             for a in a_tags]
    links[-1] = links[-1].strip('\n')
    with open(f'data/links/{category}.txt', 'w') as file:
        file.writelines(links)


def make_links(category):
    with webdriver.Chrome() as browser:
        browser.maximize_window()
        make_category(browser, category)


if __name__ == '__main__':
    os.chdir('../')
    cat_name = input()
    make_links(cat_name)
