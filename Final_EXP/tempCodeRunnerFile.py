    def process_sentence(sentence):
        words = []
        current_word = ""
        for char in sentence:
            if char.isdigit() or char in '-:.,':
                current_word += char
            elif current_word:
                if current_word.isdigit():
                    words.append(number_to_words(int(current_word)))
                else:
                    words.append(current_word)
                current_word = ""
                words.append(char)
            else:
                words.append(char)
        if current_word:
            if current_word.isdigit():
                words.append(number_to_words(int(current_word)))
            else:
                words.append(current_word)
        return ''.join(words).replace("dot dot", "dot")