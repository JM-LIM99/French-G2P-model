import pandas as pd
import editdistance
class DataProcessingGlaff():

    def __int__():
        pass


    def read_csv(self, path):
        
        df = pd.read_csv(path, sep='|', header=None, on_bad_lines='skip')
        df.columns = ['Graphie', 'Code_GRACE', 'Lemme', 'Phono_API', 'Phono_SAMPA'] + [f'Col_{i}' for i in range(5, len(df.columns))]

        return df
    
    def build_pairs(self, df):

        df = df[['Graphie', 'Phono_API']].dropna()

        graphemes = df['Graphie'].tolist()
        phonemes = [p.replace(".", "") for p in df['Phono_API'].tolist()]
        data = zip(graphemes, phonemes)

        merged_data = []
        for graph, phono in data:
            entry = {'grapheme': graph, 'phoneme': phono}
            merged_data.append(entry)

        return merged_data
    def evaluation(self, evaluation_data):
        total_errors = 0
        total_phonemes = 0
        for gold, pred in evaluation_data:
            gold = list(gold)
            pred = list(pred)

            total_errors += editdistance.eval(gold, pred)
            total_phonemes += len(gold)

        if total_phonemes == 0:
            return 0.0

        PER =total_errors/ total_phonemes
            
            
        return PER




    