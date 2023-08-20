import pickle


objects = []
with (open("lte_test100kb.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
            print(objects)
        except EOFError:
            break
