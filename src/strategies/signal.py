class Signal:

    def __init__(self, signal):
        self.signal = signal


class BuySignal(Signal):

    def __init__(self):
        super().__init__("buy")


class SellSignal(Signal):

    def __init__(self):
        super().__init__("buy")


class CloseSignal(Signal):

    def __init__(self):
        super().__init__("close")