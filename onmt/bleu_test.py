from modules.bleu import sentence_bleu_nbest

if __name__ == '__main__':
    reference = [3,1,5,4,0,0]
    hypotheses = [3,1,5,4,1,1]
    print sentence_bleu_nbest(reference, hypotheses)