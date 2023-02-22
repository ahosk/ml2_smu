from collections import Counter
import numpy as np

people = {
    "Jane": {
        "willingness to travel": 5,
        "desire for new experience": 4,
        "cost": 3,
        "indian food": 2,
        "mexican food": 5,
        "hipster points": 4,
        "vegetarian": 3,
    },
    "John": {
        "willingness to travel": 3,
        "desire for new experience": 5,
        "cost": 2,
        "indian food": 5,
        "mexican food": 4,
        "hipster points": 3,
        "vegetarian": 2,
    },
    "Bob": {
        "willingness to travel": 4,
        "desire for new experience": 5,
        "cost": 3,
        "indian food": 5,
        "mexican food": 3,
        "hipster points": 2,
        "vegetarian": 4,
    },
    "Lisa": {
        "willingness to travel": 5,
        "desire for new experience": 4,
        "cost": 2,
        "indian food": 4,
        "mexican food": 5,
        "hipster points": 5,
        "vegetarian": 6,
    },
    "Sarah": {
        "willingness to travel": 4,
        "desire for new experience": 5,
        "cost": 3,
        "indian food": 5,
        "mexican food": 4,
        "hipster points": 3,
        "vegetarian": 7,
    },
    "Tom": {
        "willingness to travel": 2,
        "desire for new experience": 3,
        "cost": 5,
        "indian food": 5,
        "mexican food": 2,
        "hipster points": 1,
        "vegetarian": 1,
    },
    "Henry": {
        "willingness to travel": 5,
        "desire for new experience": 5,
        "cost": 1,
        "indian food": 5,
        "mexican food": 5,
        "hipster points": 5,
        "vegetarian": 2,
    },
    "David": {
        "willingness to travel": 3,
        "desire for new experience": 4,
        "cost": 2,
        "indian food": 4,
        "mexican food": 3,
        "hipster points": 2,
        "vegetarian": 9,
    },
    "Emily": {
        "willingness to travel": 4,
        "desire for new experience": 5,
        "cost": 3,
        "indian food": 5,
        "mexican food": 4,
        "hipster points": 3,
        "vegetarian": 7,
    },
    "Michael": {
        "willingness to travel": 5,
        "desire for new experience": 4,
        "cost": 1,
        "indian food": 5,
        "mexican food": 5,
        "hipster points": 4,
        "vegetarian": 5,
    },
}

restaurants = {
    "Flacos": {
        "distance": 2,
        "novelty": 3,
        "cost": 2,
        "average rating": 4,
        "outside seating": 1,
        "vegetarian": 3,
        "newness": 2,
    },
    "Indian Delight": {
        "distance": 1,
        "novelty": 2,
        "cost": 3,
        "average rating": 5,
        "outside seating": 2,
        "vegetarian": 7,
        "newness": 1,
    },
    "Hipster Café": {
        "distance": 3,
        "novelty": 5,
        "cost": 4,
        "average rating": 3,
        "outside seating": 3,
        "vegetarian": 9,
        "newness": 8,
    },
    "Sushi Palace": {
        "distance": 4,
        "novelty": 4,
        "cost": 5,
        "average rating": 5,
        "outside seating": 4,
        "vegetarian": 9,
        "newness": 7,
    },
    "Steak House": {
        "distance": 5,
        "novelty": 2,
        "cost": 5,
        "average rating": 4,
        "outside seating": 5,
        "vegetarian": 1,
        "newness": 4,
    },
    "Vegetarian Haven": {
        "distance": 2,
        "novelty": 3,
        "cost": 2,
        "average rating": 5,
        "outside seating": 6,
        "vegetarian": 10,
        "newness": 9,
    },
    "Pasta Factory": {
        "distance": 3,
        "novelty": 4,
        "cost": 3,
        "average rating": 4,
        "outside seating": 9,
        "vegetarian": 7,
        "newness": 6,
    },
    "Thai Kitchen": {
        "distance": 1,
        "novelty": 5,
        "cost": 2,
        "average rating": 5,
        "outside seating": 7,
        "vegetarian": 6,
        "newness": 6,
    },
    "BBQ Joint": {
        "distance": 4,
        "novelty": 2,
        "cost": 5,
        "average rating": 4,
        "outside seating": 8,
        "vegetarian": 1,
        "newness": 2,
    },
    "Mexican Taqueria": {
        "distance": 5,
        "novelty": 3,
        "cost": 2,
        "average rating": 4,
        "outside seating": 1,
        "vegetarian": 3,
        "newness": 8,
    },
}


def create_array(dict):
    columns = list(dict[next(iter(dict))].keys())
    rows = list(dict.keys())

    matrix = []
    for i, val in enumerate(dict.values()):
        matrix.append([val[col] for col in columns])
    matrix = np.asarray(matrix)
    return columns, rows, matrix


def calc_dot_prod(**kwargs):
    matrix_1 = kwargs.get("matrix_1", None)
    matrix_2 = kwargs.get("matrix_2", None)
    if matrix_1 is None or matrix_2 is None:
        raise ValueError("Both matrix_1 and matrix_2 are required.")
    return np.dot(np.asmatrix(matrix_1), np.asmatrix(matrix_2).T)


def get_index_max(**kwargs):
    matrix_1 = kwargs.get("matrix_1", None)
    matrix_2 = kwargs.get("matrix_2", None)
    if matrix_1 is None or matrix_2 is None:
        raise ValueError("Both matrix_1 and matrix_2 are required.")
    max_dot_prod = np.argmax(calc_dot_prod(matrix_1=matrix_1, matrix_2=matrix_2))
    return max_dot_prod


def get_full_dot_prod(**kwargs):
    matrix_1 = kwargs.get("matrix_1", None)
    matrix_2 = kwargs.get("matrix_2", None)
    if matrix_1 is None or matrix_2 is None:
        raise ValueError("Both matrix_1 and matrix_2 are required.")
    full_dot_prod = calc_dot_prod(matrix_1=matrix_1, matrix_2=matrix_2)
    return full_dot_prod


def get_top_rest(person_dictionary, restaurant_list, matrix_1, matrix_2):
    top_rest_people = {}
    for i, person in enumerate(person_dictionary):
        top_rest_people.update(
            {
                person: restaurant_list[
                    get_index_max(matrix_1=matrix_1[i], matrix_2=matrix_2)
                ]
            }
        )
    return top_rest_people


def get_person_top_rest(person, dict):
    return f"{person}'s top restaurant is: {dict[person]}"


def matrix_to_dict(matrix, keys, cols):
    result = {}
    for i, key in enumerate(keys):
        row = matrix[i]
        row_dict = {}
        for j, col in enumerate(cols):
            row_dict[col] = row[j]
        result[key] = row_dict
    return result


def matrix_to_nested_dict(**kwargs):
    matrix_1 = kwargs.get("matrix_1", None)
    matrix_2 = kwargs.get("matrix_2", None)
    row_names = kwargs.get("row_names", None)
    col_names = kwargs.get("col_names", None)
    full_matrix = np.asarray(get_full_dot_prod(matrix_1=matrix_1, matrix_2=matrix_2))
    nested_dict = {}
    for i, row in enumerate(full_matrix):
        row_dict = {}
        for j, value in enumerate(row):
            row_dict[col_names[j]] = value
        nested_dict[row_names[i]] = row_dict
    return nested_dict


def sort_nested_dict(people_matrix, rest_matrix, people_rows, rest_rows):
    nested_dict = matrix_to_nested_dict(
        matrix_1=people_matrix,
        matrix_2=rest_matrix,
        row_names=people_rows,
        col_names=rest_rows,
    )
    for key in nested_dict:
        nested_dict[key] = dict(
            sorted(nested_dict[key].items(), key=lambda x: x[1], reverse=True)
        )
    return nested_dict


def find_most_common_value(input_dict):
    value_counts = Counter(input_dict.values())
    most_common_value = max(value_counts, key=value_counts.get)
    return most_common_value


def find_optimal_restaurant(nested_dict):
    optimal_restaurants = {}
    for person, restaurant_ratings in nested_dict.items():
        optimal_restaurant = max(restaurant_ratings, key=restaurant_ratings.get)
        optimal_restaurants[person] = optimal_restaurant

    return find_most_common_value(optimal_restaurants)


def update_cost_for_boss(dictionary, new_cost):
    people_list = dictionary.keys()
    for person in people_list:
        dictionary[person]["cost"] = new_cost
    return dictionary


def run(boss="FALSE"):
    global people
    global restaurants
    if boss == "FALSE":
        people_cols, people_rows, people_matrix = create_array(people)
        rest_cols, rest_rows, rest_matrix = create_array(restaurants)
        people_top_rests = get_top_rest(
            person_dictionary=people,
            restaurant_list=rest_rows,
            matrix_1=people_matrix,
            matrix_2=rest_matrix,
        )
        jane_top_rest = get_person_top_rest("Jane", people_top_rests)

        full_rest_rankings = sort_nested_dict(
            people_matrix, rest_matrix, people_rows, rest_rows
        )

        optimal_rest = find_optimal_restaurant(full_rest_rankings)
    else:
        new_cost = int(input("New price point? (1 - 10)"))
        people = update_cost_for_boss(people, new_cost)
        people_cols, people_rows, people_matrix = create_array(people)
        rest_cols, rest_rows, rest_matrix = create_array(restaurants)
        people_top_rests = get_top_rest(
            person_dictionary=people,
            restaurant_list=rest_rows,
            matrix_1=people_matrix,
            matrix_2=rest_matrix,
        )
        jane_top_rest = get_person_top_rest("Jane", people_top_rests)

        full_rest_rankings = sort_nested_dict(
            people_matrix, rest_matrix, people_rows, rest_rows
        )

        optimal_rest = find_optimal_restaurant(full_rest_rankings)

    return jane_top_rest, people_top_rests, full_rest_rankings, optimal_rest


if __name__ == "__main__":
    resp = input("Is the Boss Coming? (True/False)").upper()
    jane_top_rest, people_top_rests, full_rest_rankings, optimal_rest = run(boss=resp)
    print(
        f"{jane_top_rest}\n\nPeople Top Restaurants: \n{people_top_rests}\n\nFull Restaurant Rankings: \n{full_rest_rankings}\n\nOptimal Restaurant: \n{optimal_rest}"
    )
