from core.pipeline import Procedure
from strategies.signal import BuySignal, SellSignal


class TradingStrategy(Procedure):

    def run(self, next_diff):
        return self.execute(next_diff)

    def execute(self, next_diff):
        if next_diff > 0:
            return BuySignal()
        elif next_diff < 0:
            return SellSignal()