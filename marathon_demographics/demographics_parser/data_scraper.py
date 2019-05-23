from selenium import webdriver
import pandas as pd

driver = webdriver.Chrome(r'/Users/emre/TDI/chromedriver')
#driver.maximize_window()


names,gender_rankings,ages,nats,rankings,half_times,finish_times = [],[],[],[],[],[],[]

## London Marathon Scraper
#for i in range(1,417):
#    driver.get('http://results-2014.virginmoneylondonmarathon.com/2014/?page='+str(i)+'&event=MAS&num_results=100&pid=search&search%5Bsex%5D=%25&search%5Bnation%5D=%25&search_sort=name')
#
#    for j in range(1,101):
#        row = driver.find_element_by_xpath('//*[@id="cbox-left"]/div[5]/div[1]/table/tbody/tr[%s]'%j)
#        names.append(row.find_element_by_xpath('./td[4]/a').text[:-6])
#        gender_rankings.append(row.find_element_by_xpath('./td[2]').text)
#        ages.append(row.find_element_by_xpath('./td[7]').text)
#        nats.append(row.find_element_by_xpath('./td[4]/a').text[-4:-1])
#        rankings.append(row.find_element_by_xpath('./td[1]').text)
#        half_times.append(row.find_element_by_xpath('./td[8]').text)
#        finish_times.append(row.find_element_by_xpath('./td[9]').text)
#
#demographics = {'names':names,'ages':ages,'nats':nats,'gender_rank':gender_rankings,
#                'half_times':half_times, 'finish_times':finish_times, 'rankings':rankings}
#df = pd.DataFrame.from_dict(demographics)
#df.to_csv('./london2014.csv')


## Chicago Marathon Scraper
for i in range(1,216):
    driver.get('http://chicago-history.r.mikatiming.de/2015/?page='+str(i)+'&event=MAR_999999107FA30900000000A1&lang=EN_CAP&num_results=100&pid=list&search%5Bsex%5D=W&search%5Bage_class%5D=%25')

    for j in range(2,102):
        
        row = driver.find_element_by_xpath('//*[@id="cbox-main"]/div[2]/ul/li[%s]'%j)
        names.append(row.find_element_by_xpath('./div[1]/div/h4/a').text[:-6])
        rankings.append(row.find_element_by_xpath('./div[1]/div/div[2]').text)
        ages.append(row.find_element_by_xpath('./div[2]/div[1]/div/div[3]').text)
        nats.append(row.find_element_by_xpath('./div[1]/div/h4/a').text[-4:-1])
        gender_rankings.append(row.find_element_by_xpath('./div[1]/div/div[1]').text)
        half_times.append(row.find_element_by_xpath('./div[2]/div[2]/div/div[1]').text)
        finish_times.append(row.find_element_by_xpath('./div[2]/div[2]/div/div[2]').text)

demographics = {'names':names,'ages':ages,'nats':nats,'gender_rank':gender_rankings,
                'half_times':half_times, 'finish_times':finish_times, 'rankings':rankings}
df = pd.DataFrame.from_dict(demographics)
df.to_csv('./chicago2017.csv')

driver.quit()
