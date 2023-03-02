print("  --- THIS SCRIPT ANSWERS X QUESTIONS RELATED TO MTG AND PROBABILITIES --- ")

#%%
print(" FIRST  QUESTION ---> Total lands in starting hand depending on total lands in deck:")
from scipy.special import comb
import numpy as np


def compute_probs_initial_hand(desired_lands_in_starting_hand: int, total_land_cards: int, hand_size: int, 
                               deck_size: int = 60, verbose: bool = True) -> None:
    k, N, K, n = desired_lands_in_starting_hand, deck_size, total_land_cards, hand_size 
    probability = np.round((comb(K, k, exact=True) * comb(N-K, n-k, exact=True) * 100) / comb(N, n, exact=True), 4)
    if verbose:
        print(f" Composition of experiment ---> \nYou want {desired_lands_in_starting_hand} lands in {hand_size} cards starting hand")
        print(f" With a total of {total_land_cards} land cards in your {deck_size} cards deck \n")
        print(f" Resulting probability is ---> {probability}%")
    return probability



for lands in [21, 22, 23]:
    p = compute_probs_initial_hand(3, lands, 7, 60, False)
    print("Prob to begin with 3 lands (%) --> ", p)


#%%
print(" SECOND QUESTION ---> Probability of having at least 1 land in hand. It should be equal to 1 minus 0 lands in initial hand: ")

print("Using Hypergeometric dist: ")
total_land_cards = 22
desired_lands_in_starting_hand = 0
no_lands_prob = compute_probs_initial_hand(desired_lands_in_starting_hand, total_land_cards, 7, 60, False)
print(f" Prob of 0 lands hand: {100-no_lands_prob}%")


print("Computing probabilities of the rest of the scenarios: ")
prob_at_least_one_land = 0
desired_lands_in_starting_hand = 0

for i in range(7):
    desired_lands_in_starting_hand = desired_lands_in_starting_hand + 1
    prob_at_least_one_land += compute_probs_initial_hand(desired_lands_in_starting_hand, total_land_cards, 7, 60, False)

    
print(f"Iteratively way of computing at least 1 land probability (%) {prob_at_least_one_land}")
    
#%%
print(" THIRD QUESTION ---> Probability of having 3 lands on the third turn: ")
print(" Using my function \n")

print(" ON THE PLAY  + 1 draw T2 + 1 draw T3: ")
compute_probs_initial_hand(3, 22, 9, 60)
print(" \n ON THE DRAW + 1 draw T1  + 1 draw T2 + 1 draw T3: ")
compute_probs_initial_hand(3, 22, 10, 60)


print(" Using scipy's function \n")
from scipy.stats import hypergeom
import matplotlib.pyplot as plt



total_cards, total_lands = 60, 22

otp = hypergeom(total_cards, total_lands, 10)
otd = hypergeom(total_cards, total_lands, 9)

lands_in_hand = np.arange(10)      # 0 to 9 lands 
prob_no_start = otd.pmf(lands_in_hand)
prob_start = otp.pmf(lands_in_hand)

plt.plot(lands_in_hand, prob_start * 100, 'bo', label='on_the_play')
plt.plot(lands_in_hand, prob_no_start * 100, 'go', label='on_the_draw')
plt.title(f"Land probalities (%) on T3 with {total_lands} lands being OTP or OTD")
plt.xticks(lands_in_hand)
plt.legend()
plt.show()
print(" The first result may not be 100% intuitive. with 1 more draw you will have 4 lands on T3 probably")


#%%
import random
print("Simulacion")

deck_size = 60
remaining_lands = 23
no_lands = deck_size - remaining_lands
hand_size = 7


def simulate_hands(hands_to_simulate: int, lands_in_deck: int, hand_size: int = 7,
                   deck_size: int = 60) -> list[list]:
    hands_value_list, hands_rep_list = list(), list()
    for i in range(hands_to_simulate):
        remaining_lands = lands_in_deck
        # hand_values, hand_reps = list(), list() 
        hand_value, hand_rep = 0, ""
        remaining_deck_size = deck_size
        for i in range(hand_size):
            # print(remaining_deck_size)
            land_p = remaining_lands / remaining_deck_size
            land_coin_toss = random.uniform(0, 1)
            if land_coin_toss < land_p:
                remaining_lands -= 1
                hand_value += 1
                hand_rep += "L"
            else:
                hand_rep += "S"
            remaining_deck_size -= 1
        hands_value_list.append(hand_value)
        hands_rep_list.append(hand_rep)
    return hands_value_list, hands_rep_list



import pandas as pd
import matplotlib.pyplot as plt
# pd.Series(hands_value_list).hist()

n_simulations = 1000000
hands_value_list_draw, hands_rep_list_draw = simulate_hands(hands_to_simulate=n_simulations, 
                lands_in_deck=22, hand_size=10)

hands_value_list_play, hands_rep_list_play = simulate_hands(hands_to_simulate=n_simulations, 
                lands_in_deck=22, hand_size=9)

df_results = pd.DataFrame({"hand_value_play": hands_value_list_play, "hand_rep_play": hands_rep_list_play,
                           "hand_value_draw": hands_value_list_draw, "hand_rep_draw": hands_rep_list_draw})

# sns.kdeplot(data=df_results, x="hand_value_play")
# sns.kdeplot(data=df_results, x="hand_value_draw")
title = "Distribution of lands in "  + str(n_simulations) + " simulated hands"  

land_draw_prob = [compute_probs_initial_hand(lands, 22, 10, 60, False)/100 for lands in range(11)]

land_play_prob = [compute_probs_initial_hand(lands, 22, 9, 60, False)/100 for lands in range(10)]

# sns.kdeplot(data=df_results[["hand_value_play", "hand_value_draw"]]).set_title(title)

df_hand_otp = (df_results["hand_value_play"].value_counts()/n_simulations).to_frame().sort_index()
df_hand_otp["theoretical_value"] = land_play_prob
df_hand_otd = (df_results["hand_value_draw"].value_counts()/n_simulations).to_frame().sort_index()
df_hand_otd["theoretical_value"] = land_draw_prob



fig, axes = plt.subplots(2, 1)
plt.suptitle("Land number probabilities", fontsize=14)
df_hand_otp.plot.bar(ax=axes[0])
df_hand_otd.plot.bar(ax=axes[1])


#%%

def compute_prob_N_lands_T_turn(desired_lands: int, turn: int, 
                                lower_bound_lands: int, upper_bound_lands) -> None:  
    print(f"You want to know the chances of having exactly {desired_lands} lands in turn {turn}")
    for i in range(lower_bound_lands, upper_bound_lands + 1):
        p = compute_probs_initial_hand(desired_lands, i, turn + 7, 60, False)
        print(f" For {i} lands in deck {p}")
    
compute_prob_N_lands_T_turn(3, 3, 20, 23)

#%%
print(" If you begin with 1 Land initial 7 cards hand ")
next_draw_land = compute_probs_initial_hand(1, 21, 1, 53, False)
print(" Probabilty of next draw land ", next_draw_land)
print(" Which is the same as computing: ", 21/53, "\n")

next_two_draws_land = compute_probs_initial_hand(2, 21, 2, 53, False)
print(" Probabilty of next 2 draws are land ", next_two_draws_land)
print(f" Which indeed is the same as computing  21/53 *20/52 = {np.round(21/53 *20/52, 4)}")























