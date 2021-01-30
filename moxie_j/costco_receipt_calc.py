# costco receipt calculator
import numpy as np


def main():
    prices = {'ks_cold_brew': 13.99, 'kokoho_rice': 24.99, 'amlactin': 17.59,
              'olive_oil': 19.99, 'whole_milk': 2.47, 'simple_green': 9.49,
              'san_marzano': 8.99, 'brussel_sprt': 3.49, 'bagels': 6.99,
              'org_bellas': 4.79, 'choc_crepes': 6.99, 'barilla_thin': 8.99,
              'org_silk': 8.99, 'tble_salt': 1.39, 'ks_food_wrap': 9.89,
              'fryer_thighs': 12.26, 'org_spinach': 3.99, 'lactaid': 4.36,
              'mocha_frap': 16.99, '40pk_aa': 16.99, 'fryer_thighs2': 10.45
              }

    tax_stat = {'ks_cold_brew': 'E', 'kokoho_rice': 'E', 'amlactin': 'F',
                'olive_oil': 'E', 'whole_milk': 'E', 'simple_green': 'A',
                'san_marzano': 'E', 'brussel_sprt': 'E', 'bagels': 'E',
                'org_bellas': 'E', 'choc_crepes': 'E', 'barilla_thin': 'E',
                'org_silk': 'E', 'tble_salt': 'E', 'ks_food_wrap': 'A',
                'fryer_thighs': 'E', 'org_spinach': 'E', 'lactaid': 'E',
                'mocha_frap': 'E', '40pk_aa': 'A', 'fryer_thighs2': 'E'
                }

    ### USER INPUT SECTION: REPLACE WITH YOUR OWN QUANTITIES
    all_items = list(prices.keys())
    all_items_quant = dict.fromkeys(all_items, 1)
    all_items_quant['lactaid'] = 3
    all_items_quant['san_marzano'] = 2
    all_items_quant['amlactin'] = 2

    justin_items = all_items.copy()
    justin_items.remove('lactaid')
    justin_items.remove('mocha_frap')
    justin_items.remove('40pk_aa')
    justin_items_quant = all_items_quant.copy()
    justin_items_quant['amlactin'] = 1
    del justin_items_quant['lactaid']
    del justin_items_quant['mocha_frap']
    del justin_items_quant['40pk_aa']
    ### END OF USER INPUT SECTION

    running_sum = 0
    for item in all_items:
        price = prices[item]
        tax_code = tax_stat[item]
        quantity = all_items_quant[item]
        print('item:',item,'price:',price,'tax code:',tax_code,'quantity:',quantity)
        running_sum = scan_item(running_sum, price, tax_code, quantity)
    print('total cost is:', running_sum,'\n')

    running_sum = 0
    for item in justin_items:
        price = prices[item]
        tax_code = tax_stat[item]
        quantity = justin_items_quant[item]
        print('item:', item, 'price:', price, 'tax code:', tax_code, 'quantity:', quantity)
        running_sum = scan_item(running_sum, price, tax_code, quantity)
    print('justin\'s cost is:', running_sum)

    # debug
    # print(all_items_quant)


def scan_item(sum, price, tax_code, quantity):
    if tax_code == 'E':
        rate = 1.0
    elif tax_code == 'A':
        rate = 1.06
    elif tax_code == 'F':
        rate = 1.06
    else:
        print('tax code not recognized, defaulting to 1.0')
        rate = 1.0

    sum += price*quantity*rate
    return sum

if __name__ == "__main__":
    main()
