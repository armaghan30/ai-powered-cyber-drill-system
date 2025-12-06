class TinyActions:

    def __init__(self, hosts):
        self.hosts = hosts
        self.action_list = []

        # SCAN
        for h in hosts:
            self.action_list.append({"type": "scan", "target": h})

        # EXPLOIT
        for h in hosts:
            self.action_list.append({"type": "exploit", "target": h})

        # PRIVESC
        for h in hosts:
            self.action_list.append({"type": "privesc", "target": h})

        # PIVOT
        for h in hosts:
            self.action_list.append({"type": "pivot", "target": h})

        # FLAG (final)
        self.action_list.append({"type": "flag", "target": "H4"})

        # NO-OP
        self.action_list.append({"type": "noop", "target": None})

    def get_action(self, idx):
        return self.action_list[idx]

    def count(self):
        return len(self.action_list)
