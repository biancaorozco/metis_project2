#####################################
##########ExpediaScraper.py##########
#####################################
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException

import urllib.robotparser
import pandas as pd
import time
import datetime

debug = True


############### User Agent ###############
def get_user_agent():
    
    '''
    returns:
    Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) 
    AppleWebKit/537.36 (KHTML, like Gecko) 
    Chrome/77.0.3865.90 Safari/537.36'
    '''
    driver = webdriver.Chrome()

    #Store agent in a variable and print the value
    agent = driver.execute_script("return navigator.userAgent;")
    return agent

agent = get_user_agent()


############### Webdriver ###############
opts = Options()
opts.add_argument(agent)
browser = webdriver.Chrome(executable_path = 'chromedriver', chrome_options = opts)
browser.implicitly_wait(10)


############### Travel Services ###############
# Options
flights = "//button[@id='tab-flight-tab-hp']"
hotels = "//button[@id='tab-hotel-tab-hp']"
cars = "//button[@id='tab-car-tab-hp']"
vacay_rentals = "//button[@id='tab-vacation-rental-tab-hp']"

# Choose Travel Service
def choose_service(service):
    
    service_type = browser.find_element_by_xpath(service)
    
    service_type.click()


############### Flights ###############
# Options
roundtrip = "//label[@id='flight-type-roundtrip-label-hp-flight']"
one_way = "//label[@id='flight-type-one-way-label-hp-flight']"
multi_city = "//label[@id='flight-type-multi-dest-label-hp-flight']"

# Choose Flight Option
def choose_flight(flight):
    
    flight_type = browser.find_element_by_xpath(flight)
    
    flight_type.click()


############### Cities ###############
list_of_cities = ["Atlanta ATL", "Los Angeles LAX", "Chicago ORD", "Dallas DFW", "Denver DEN", "New York JFK", "San Francisco SFO", "Seattle SEA"]#, "Las Vegas LAS", "Orlando MCO", "Charlotte CLT", "Phoenix PHX", "Houston IAH", "Miami MIA", "Boston BOS", "Minneapolis MSP", "Fort Lauderdale FLL", "Detroit DTW", "Philadelphia PHL", "Baltimore BWI", "Salt Lake City SLC", "San Diego SAN", "Washington, D.C. IAD", "Tampa TPA", "Portland PDX", "Honolulu HNL"]

# Shuffle Cities List
def shuffle_cities(list_of_cities):
    list_len = len(list_of_cities)
    print(list_len)
    assert list_len % 2 == 0
    assert len(list_of_cities) == len(set(list_of_cities))
    combined_lists = []

    first_half = list_of_cities[:list_len//2]
    second_half = list_of_cities[list_len//2:]

    for i in range(list_len - 2):
        last_item_first_half = first_half.pop(-1)
        first_item_second_half = second_half.pop(0)

        first_half.insert(1, first_item_second_half)
        second_half.append(last_item_first_half)

        combined_lists.append(list(zip(first_half, second_half)))
    return combined_lists

# Execute
city_pairs = shuffle_cities(list_of_cities)


############### Departure City ###############
def select_departure_city(departure_city):
    
    try:
        fly_from = browser.find_element_by_xpath("//input[@id='flight-origin-hp-flight']")
        fly_from.clear()
        fly_from.send_keys(' ' + departure_city)
        
    except StaleElementReferenceException:
        pass
    
    # Selects first suggestion from auto-complete drop-down menu
    try:
        first_item = browser.find_element_by_xpath("//a[@id='aria-option-0']")
        first_item.click()
        
    except StaleElementReferenceException:
        pass


############### Arrival City ###############
def select_arrival_city(arrival_city):
    try:
        fly_to = browser.find_element_by_xpath("//input[@id='flight-destination-hp-flight']")
        fly_to.clear()
        fly_to.send_keys(' ' + arrival_city)
    except StaleElementReferenceException:
        pass

    
    # Selects first suggestion from auto-complete drop-down menu
    try:
        first_item = browser.find_element_by_xpath("//a[@id='aria-option-0']")
        first_item.click()
    except StaleElementReferenceException:
        pass


############### Date ###############
day = '8'
month = '10'
year = '2019'

# Input Departure Date
def choose_departure_date(day, month, year):
    dept_date_button = browser.find_element_by_xpath("//input[@id='flight-departing-single-hp-flight']")
    dept_date_button.clear()
    dept_date_button.send_keys(month + '/' + day + '/' + year)


############### Click Search ###############
def select_search():
    search = browser.find_element_by_xpath("//button[@class='btn-primary btn-action gcw-submit']")
    search.submit()
    


############### Data Frame ###############
df = pd.DataFrame()
def combine_data(day, month, year, from_city, to_city):
    global df
    global departure_times_list
    global arrival_times_list
    global airlines_list
    global durations_list
    global stops_list
    global airport_info_list
    global prices_list

    # adding new rows after existing ones
    csv_index = len(df.index)

    # Departure Times
    try:
        departure_times = browser.find_elements_by_xpath("//span[@data-test-id='departure-time']")
        departure_times_list = [value.text for value in departure_times]
    except StaleElementReferenceException:
        pass

    # Arrival Times
    try:
        arrival_times = browser.find_elements_by_xpath("//span[@data-test-id='arrival-time']")
        arrival_times_list = [value.text for value in arrival_times]
    except StaleElementReferenceException:
        pass

    # Airline
    try:
        airlines = browser.find_elements_by_xpath("//span[@data-test-id='airline-name']")
        airlines_list = [value.text for value in airlines]
    except StaleElementReferenceException:
        pass

    # Duration
    try:
        durations = browser.find_elements_by_xpath("//span[@data-test-id='duration']")
        durations_list = [value.text for value in durations]
    except StaleElementReferenceException:
        pass

    # Stops
    try:
        stops = browser.find_elements_by_xpath("//span[@class='number-stops']")
        stops_list = [value.text for value in stops]
    except StaleElementReferenceException:
        pass

    # Price
    try:
        prices = browser.find_elements_by_xpath("//span[@data-test-id='listing-price-dollars']")
        prices_list = [value.text for value in prices]
    except StaleElementReferenceException:
        pass


# I understand this code!!!!
    for i in range(len(departure_times_list)): # This is just for the sake of getting length of rows
        # First column: Numbering flight data
        try:
            df.loc[int(i) + csv_index, 'id_flight'] = int(i)
        except Exception as e:
            pass
        # Second column: Departure Times
        try:
            df.loc[i + csv_index, 'departure_time'] = departure_times_list[i]
        except Exception as e:
            pass
        # Third column: Arrival Times
        try:
            df.loc[i + csv_index, 'arrival_time'] = arrival_times_list[i]
        except Exception as e:
            pass
        # Fourth column: Airline
        try:
            df.loc[i + csv_index, 'airline'] = airlines_list[i]
        except Exception as e:
            pass
        # Fifth column: Flight Duration
        try:
            df.loc[i + csv_index, 'duration'] = durations_list[i]
        except Exception as e:
            pass
        # Sixth column: Number of Stops
        try:
            df.loc[i + csv_index, 'number_stops'] = stops_list[i]
        except Exception as e:
            pass
        # Seventh column: Departure Airport
        try:
            df.loc[i + csv_index, 'departure_airport'] = from_city
        except Exception as e:
            pass
        # Eigth column: Arrival Airport
        try:
            df.loc[i + csv_index, 'arrival_airport'] = to_city
        except Exception as e:
            pass
        # Ninth column: Prices
        try:
            df.loc[i + csv_index, 'prices'] = prices_list[i]
        except Exception as e:
            pass    
                # Ninth column: Prices
        try:
            df.loc[i + csv_index, 'date'] = month + '/' + day + '/' + year
        except Exception as e:
            pass   
    

############### Execute ###############
parser = urllib.robotparser.RobotFileParser()
parser.set_url("https://www.expedia.com/robots.txt")
parser.read()

if parser.can_fetch("*", "https://www.expedia.com/Flights"):

    for i in range(0,len(city_pairs)): # length = 24 lists
        for j in range(0,len(city_pairs[0])): # length = 13 pairs in each list
            link = 'https://www.expedia.com/'
            browser.get(link)
            choose_service(flights)
            choose_flight(one_way)
            from_city = ''
            to_city = ''
            for k in (0,len(city_pairs[0][0])): # length = 2 cities in each pair
            
                
                if k == 0:
                    from_city = city_pairs[i][j][k]
                    select_departure_city(from_city)
                    
                else:
                    to_city = city_pairs[i][j][1]
                    select_arrival_city(to_city)
                    choose_departure_date(day, month, year)
                    select_search()

            combine_data(day, month, year, from_city, to_city)

            if (debug):
                print(df)
                
    df.to_csv('expedia_' + day + '_'+ month + '_' + year +'.csv', header=True, index=False)
else:
    print('Error: No permission to extract this data')

