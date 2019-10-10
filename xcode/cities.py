def shuffle_cities(list_of_cities):
    list_len = len(list_of_cities)
    assert list_len % 2 == 0
    assert len(list_of_cities) == len(set(list_of_cities))
    combined_lists = []

    first_half = list_of_cities[:list_len//2]
    second_half = list_of_cities[list_len//2:]

    # Sanity Check
    #combined_lists = (list(zip(first_half, second_half)))
    # print("Original", combined_lists)

    for i in range(list_len - 2):
        last_item_first_half = first_half.pop(-1)
        first_item_second_half = second_half.pop(0)

        first_half.insert(1, first_item_second_half)
        second_half.append(last_item_first_half)

        combined_lists.append(list(zip(first_half, second_half)))
    return combined_lists

# Source for major airports: https://www.world-airport-codes.com/us-top-40-airports.html
list_of_cities = ["Atlanta ATL", "Los Angeles LAX", "Chicago ORD", "Dallas DFW", "Denver DEN", "New York JFK", "San Francisco SFO", "Seattle SEA", "Las Vegas LAS", "Orlando MCO", "Charlotte CLT", "Phoenix PHX", "Houston IAH", "Miami MIA", "Boston BOS", "Minneapolis MSP", "Fort Lauderdale FLL", "Detroit DTW", "Philadelphia PHL", "Baltimore BWI", "Salt Lake City SLC", "San Diego SAN", "Washington, D.C. IAD", "Tampa TPA", "Portland PDX", "Honolulu HNL"]

# Execute
city_pairs = shuffle_cities(list_of_cities)