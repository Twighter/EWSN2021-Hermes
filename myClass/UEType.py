class UEType:
    def __init__(self, rbNum, rbgNum):
        self.dist = 0
        self.maxCQI = 0
        self.cqi = []

        self.buffer = 0
        self.history = []

        self.app = []

        self.throughput = []
