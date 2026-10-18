"""Privacy-Utility Tradeoff Calculator (Eq. 5, Section III.C.2)."""
class PUTCEngine:
    def __init__(self, alpha=0.6, beta=0.4):
        self.alpha, self.beta = alpha, beta

    def compute(self, utility, privacy_cost):
        return self.alpha * utility - self.beta * privacy_cost