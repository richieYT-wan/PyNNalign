from src.data_processing import encode_batch


def main():
    try:
        seqs = ['ARNARNARNARN', 'ARNARNARNARN','ARNARNARNARN','ARNARNARNARN']
        encode_batch(seqs)
    except:
        raise Exception('There')
    print('here')

if __name__=='__main__':
    main()