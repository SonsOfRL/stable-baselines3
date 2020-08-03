
class A():

    def __init__(self):
        print("A")


class B(A):

    def __init__(self):
        print("B")
        super().__init__()


class C(B):

    def __init__(self):
        print("C")
        super(B, self).__init__()


if __name__ == "__main__":
    C()
