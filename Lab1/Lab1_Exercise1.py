'''Exercise 1 - Managing text and filtering
Using the following words:
[" further ", " Forward ", " Foreign ", " financE ", " Forgive ", " feature ", " federal ",
" failurE ", " Feeling ", " finding ", " freedom ", " Foundry "]
(a) Print all words beginning with fo
(b) Print all words ending with e
(c) Reflect on how simple techniques like these can bring value to a project
(d) Converting words to lowercase is a frequent process in text cleaning. Can you think of any issues
that arise by doing this?
'''

def print_words_beginning_with_fo(array):
    # Remove whitespace and make characters lowercase
    clean_words = remove_whitespace_and_make_lowercase(array)
    for word in clean_words:
        if word[:2] == "fo":
            print(word)


def remove_whitespace_and_make_lowercase(array):
    return [w.lower().strip() for w in array]


def print_words_ending_with_e(array):
    clean_words = remove_whitespace_and_make_lowercase(array)
    for word in clean_words:
        if word[-1] == 'e':
            print(word)


if __name__ == '__main__':
    words = [" further ", " Forward ", " Foreign ", " financE ", " Forgive ", " feature ", " federal ",
             " failurE ", " Feeling ", " finding ", " freedom ", " Foundry "]

    print("\n\nPrinting words beginning with 'fo':")
    print_words_beginning_with_fo(words)

    print("\n\nPrinting words ending with 'e':")
    print_words_ending_with_e(words)