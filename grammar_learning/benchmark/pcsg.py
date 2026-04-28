import random
import nltk
from nltk.grammar import ProbabilisticProduction, Nonterminal
import heapq
from collections import deque


def from_pcsg_string(grammar_string, start_symbol="S"):
    """
        Convert a string representation of a PCFG into a PCFG object.
    """

    productions = []
    nonterminals = set()
    for line in grammar_string.split('\n'):
        line = line.strip()
        if "->" not in line:
            continue



        lhs_string, rhs_prob_string = line.split("->")
        
        # parse lhs
        lhs = []
        lhs_string = lhs_string.strip()
        for elem in lhs_string.split():
            # terminal must start with '
            if elem.startswith("'"):
                lhs.append(elem.replace("'", ""))
            else:
                assert elem[0].isupper(), f"Nonterminal {elem} must start with a capital letter"
                lhs.append(Nonterminal(elem))
                if Nonterminal(elem) not in nonterminals:
                    nonterminals.add(Nonterminal(elem))
        # print(lhs)
        # print([type(elem) for elem in lhs])




        assert "[" in rhs_prob_string, f"Expected '[' in {rhs_prob_string} to denote probability"
        rhs_string, prob_string = rhs_prob_string.strip().split("[")
        
        # parse rhs
        rhs = []
        rhs_string = rhs_string.strip()
        for elem in rhs_string.split():
            # terminal must start with '
            if elem.startswith("'"):
                rhs.append(elem.replace("'", ""))
            else:
                assert elem[0].isupper(), f"Nonterminal {elem} must start with a capital letter"
                rhs.append(Nonterminal(elem))
                if Nonterminal(elem) not in nonterminals:
                    nonterminals.add(Nonterminal(elem))
        # print(rhs)
        # print([type(elem) for elem in rhs])
        

        # parse probability
        assert "]" in prob_string
        prob_string = prob_string.strip()[:-1]
        prob = float(prob_string)
        
        
        # print(rhs_string)
        # print(prob)


        productions.append(
            ProbabilisticProduction(tuple(lhs), tuple(rhs), prob=prob)
        )

    # print(nonterminals)
    assert Nonterminal(start_symbol) in nonterminals, f"Start symbol {start_symbol} not in nonterminals"    
    return PCSG(productions, Nonterminal(start_symbol))



class PCSG:
    def __init__(self, productions, start_symbol):
        """
        Initializes a probabilistic context-sensitive grammar (PCSG).
        
        Args:
            productions (list): List of ProbabilisticProduction rules.
            start_symbol (Nonterminal): The start symbol of the grammar.
        """
        self.productions = productions
        self.start_symbol = start_symbol
        self._lhs_index = self._index_lhs_productions()
        self._lexical_index = self._get_terminals()


    def __str__(self):
        
        s = f"PCSG with {len(self.productions)} productions:\n"
        for production in self.productions:
            s += f"{production}" + "\n"
        return s
    

    def _get_terminals(self):
        terminals = {}
        for production in self.productions:
            for elem in production.rhs():
                if isinstance(elem, str):
                    terminals[elem] = True
            for elem in production.lhs():
                if isinstance(elem, str):
                    terminals[elem] = True

        return terminals


    def _index_lhs_productions(self):
        """
        Index productions based on their left-hand side (LHS).
        Returns a dictionary mapping LHS symbols to production rules.
        """
        index = {}
        probs = {}
        self.nonterminals = {}
        for production in self.productions:
            assert len(production.lhs()) >= 1
            assert len(production.rhs()) >= len(production.lhs())
            lhs = tuple(production.lhs())  # LHS can be a tuple in CSG
            if lhs not in index:
                index[lhs] = []
            if lhs not in probs:
                probs[lhs] = []
            index[lhs].append(production)
            probs[lhs].append(production.prob())
            for nt in production.lhs():
                if nt not in self.nonterminals:
                    self.nonterminals[nt] = True

        self.nonterminals = set(self.nonterminals.keys())
        for key in probs:
            assert abs(sum(probs[key]) - 1) <= 0.03, f"Probabilities for {key} do not sum to 1: {probs[key]} | {sum(probs[key])}"
        return index

    def generate(self, n, seed=None):
        """
        Generates `n` strings from the PCSG.
        
        Args:
            n (int): Number of sentences to generate.
            seed (int, optional): Random seed.
        
        Yields:
            tuple: Generated sentence and its probability.
        """
        random.seed(seed)
        for _ in range(n):
            self.lhs_applied_position = {}
            yield self._generate_derivation([self.start_symbol]), self.lhs_applied_position
            # yield self._generate_derivation([self.start_symbol])

    def _choose_production_reducing(self, context):
        """
        Selects a production rule that can be applied given the context.
        
        Args:
            context (list): Current sentence state.
        
        Returns:
            tuple: (Production rule, probability, index in context).
        """
        applicable_rules = []
        for length in range(len(context), 0, -1): # prioritize longer segments
            found_one = False
            for i in range(len(context) - length + 1):
                assert i + length <= len(context)
                segment = tuple(context[i:i+length])
                if segment in self._lhs_index:
                    productions = self._lhs_index[segment]
                    probabilities = [prod.prob() for prod in productions]
                    applicable_rules.append((productions, probabilities, i, length)) # store all information. Sample later
                    found_one = True
            if found_one: # prioritize longer segments
                break


        if not applicable_rules:
            return None  # No valid rule to apply

        # if len(applicable_rules) >= 1:
        #     print(context)
            # for applicable_rule in applicable_rules:
            #     print(applicable_rule)
            

        # applicable_rule = random.choice(applicable_rules) # random
        applicable_rule = applicable_rules[0] # deterministic (left to right)
        productions = applicable_rule[0]
        production_indices = list(range(len(productions)))
        probabilities = applicable_rule[1]
        i = applicable_rule[2]
        length = applicable_rule[3]

        # random select production index
        random_production_index = random.choices(production_indices, weights=probabilities)[0]
        return productions[random_production_index], probabilities[random_production_index], i, length, random_production_index
        # return random.choice(applicable_rules)

    def _generate_derivation(self, sentence):
        """
        Expands the sentence until only terminal symbols remain.
        
        Args:
            sentence (list): Current sentence state.
        
        Returns:
            tuple: (Generated sentence, total probability).
        """
        probability = 1.0
        while any(isinstance(sym, Nonterminal) for sym in sentence):
            choice = self._choose_production_reducing(sentence)
            if choice is None:
                break
            

            production, prob, index, length, production_rule_index = choice
            probability *= prob
            sentence[index:index+length] = production.rhs()  # Apply production
            if production.lhs() not in self.lhs_applied_position:
                self.lhs_applied_position[production.lhs()] = []
            self.lhs_applied_position[production.lhs()].append((index, 
                                                                index+len(production.rhs()),
                                                                (0, production.lhs(), production_rule_index)))
        
        return tuple(sentence), probability


    
    # # an expensive parsing

    # def _index_rhs_productions(self):
    #     """
    #     Index productions based on their right-hand side (RHS).
    #     Returns a dictionary mapping RHS tuples to lists of production rules.
    #     This is used for reverse (parsing) search: RHS -> LHS.
    #     """
    #     index = {}
    #     for production in self.productions:
    #         rhs = tuple(production.rhs())
    #         if rhs not in index:
    #             index[rhs] = []
    #         index[rhs].append(production)
    #     return index


    # def parse(self, tokens, max_steps=100000, node_limit=50000, return_derivation=False):
    #     """
    #     Brute-force parser for the PCSG (context-sensitive grammar) using
    #     reverse rewriting: starting from the terminal sentence or tokens, repeatedly
    #     apply inverse productions (RHS -> LHS) to see if we can reach
    #     the start symbol.

    #     Args:
    #         tokens (tuple(str)): Input sentence as a tuple of terminals.
    #         max_steps (int): Hard cap on total exploration steps to avoid
    #                          infinite / huge searches.
    #         return_derivation (bool): If True, also return one derivation
    #                                   (sequence of applied productions) and
    #                                   its probability.

    #     Returns:
    #         If return_derivation == False:
    #             bool: True if accepted by the grammar, False otherwise.
    #         If return_derivation == True:
    #             (accepted, prob, derivation)
    #             where:
    #                 accepted (bool)
    #                 prob (float): probability of *one* found derivation
    #                 derivation (list): list of steps (index, production)
    #                                    in reverse order (sentence -> start)
    #     """
        
    #     assert isinstance(tokens, tuple)

    #     # Build RHS index if not already available
    #     if not hasattr(self, "_rhs_index"):
    #         self._rhs_index = self._index_rhs_productions()

    #     # Priority queue: items are (-prob, steps_so_far, current_seq, derivation)
    #     # We use -prob so that higher probability is popped first.
    #     start_state = tokens
    #     pq = []
    #     heapq.heappush(pq, (-1.0, 0, start_state, []))

    #     visited = set()
    #     visited.add(start_state)

    #     target = (self.start_symbol,)
    #     total_popped = 0
    #     steps = 0

    #     while pq and steps < max_steps and total_popped < node_limit:
    #         steps += 1
    #         neg_prob, depth, current, derivation = heapq.heappop(pq)
    #         total_popped += 1
    #         prob = -neg_prob

    #         # --- Acceptance condition ---
    #         if current == target:
    #             if return_derivation:
    #                 return True, prob, derivation
    #             else:
    #                 return True

    #         n = len(current)

    #         # --- Try all spans for reverse rewriting (RHS -> LHS) ---
    #         for i in range(n):
    #             for length in range(1, n - i + 1):
    #                 segment = tuple(current[i:i+length])

    #                 if segment in self._rhs_index:
    #                     for prod in self._rhs_index[segment]:
    #                         # Replace this span with the production's LHS
    #                         new_seq = current[:i] + tuple(prod.lhs()) + current[i+length:]

    #                         if new_seq not in visited:
    #                             visited.add(new_seq)
    #                             new_prob = prob * prod.prob()
    #                             new_derivation = derivation + [(i, prod)]
    #                             # Depth increases by 1 for each rewrite
    #                             heapq.heappush(pq, (-new_prob, depth + 1, new_seq, new_derivation))

    #     # If we exit the loop: either queue is empty or we hit limits
    #     if return_derivation:
    #         return False, 0.0, None
    #     else:
    #         return False

    #     # # BFS queue: each item is (current_sequence, prob, derivation_steps)
    #     # # current_sequence: tuple of symbols (Nonterminals and/or strings)
    #     # # prob: product of production probabilities along this path
    #     # # derivation_steps: list of (position, production) used in reverse
    #     # start_state = tokens
    #     # queue = deque()
    #     # queue.append((start_state, 1.0, []))

    #     # visited = set()
    #     # visited.add(start_state)

    #     # steps = 0
    #     # target = (self.start_symbol,)  # we want to reduce to just the start symbol

    #     # while queue and steps < max_steps:
    #     #     current, prob, derivation = queue.popleft()
    #     #     steps += 1

    #     #     # Acceptance condition
    #     #     if current == target:
    #     #         if return_derivation:
    #     #             return True, prob, derivation
    #     #         else:
    #     #             return True

    #     #     n = len(current)

    #     #     # Try applying inverse productions on all spans
    #     #     for i in range(n):
    #     #         for length in range(1, n - i + 1):
    #     #             segment = tuple(current[i:i+length])

    #     #             # If this span matches any RHS, we can reduce it to the LHS
    #     #             if segment in self._rhs_index:
    #     #                 for prod in self._rhs_index[segment]:
    #     #                     # Replace this span with the production's LHS
    #     #                     new_seq = current[:i] + tuple(prod.lhs()) + current[i+length:]

    #     #                     if new_seq not in visited:
    #     #                         visited.add(new_seq)
    #     #                         new_prob = prob * prod.prob()
    #     #                         new_derivation = derivation + [(i, prod)]
    #     #                         queue.append((new_seq, new_prob, new_derivation))

    #     # # If we exhaust the queue or hit max_steps without finding the start symbol
    #     # if return_derivation:
    #     #     return False, 0.0, None
    #     # else:
    #     #     return False

