import pywrapfst as fst


import sys

def get_word_id(isym, word):
    word_id = isym.find(word)
    if word_id<0:
        print(f"'{word}' missing in lexicon. Cannot find valid word id. Fix lexicon and try again")
        exit(1)
    else:
        return word_id


def create_tag_fst(isym, tag_filename):
    with open(tag_filename,'r') as tfile:
        tag_fst = fst.VectorFst()
        entity_name = f'{tag_filename.split("/")[-1].split(".")[0]}'
        tag_name = f"#entity:{entity_name}"

        tag_name_id = isym.find(tag_name)
        if tag_name_id<0:
            print(f"Missing nonterm symbol for {tag_name}")
            exit(1)

        eps=0  # Check words.txt if this is different
        #Need to have a better way to add weights (sub-language modeling?)
        default_weight=fst.Weight(tag_fst.weight_type(),-0.01)
        tag_state=tag_fst.add_state()
        tag_fst.set_start(tag_state)
        # Skip adding the start state #entity:<entity_name> since this is handled by fstreplace
        #tag_new_state=tag_fst.add_state()
        #tag_fst.add_arc(tag_state, fst.Arc(tag_name_id, eps, eps_weight, tag_new_state))
        phrase_start_state=tag_state
        phrase_end_state=tag_fst.add_state()

        history_grammar=[]
        for line in tfile:
            if line.strip() in history_grammar:
                print("Duplicate grammar, skipping")
                continue
            history_grammar.append(line.strip())
            words = line.strip().split(" ")
            state=phrase_start_state
            for word in words[:-1]:
                word_id=get_word_id(isym,word)
                if word_id > 0:

                    new_state=tag_fst.add_state()
                    tag_fst.add_arc(state,fst.Arc(word_id,word_id,default_weight,new_state))
                    state = new_state
                else:
                    print(f"Could not find {word} in symbol table")
            word_id = get_word_id(isym,words[-1])

            tag_fst.add_arc(state, fst.Arc(word_id, word_id, default_weight, phrase_end_state))

        tag_final_state=tag_fst.add_state()
        tag_fst.add_arc(phrase_end_state, fst.Arc(tag_name_id, eps, default_weight, tag_final_state))
        tag_fst.set_final(tag_final_state, default_weight)
    #tag_fst.write("/tmp/tag_nomin.fst")
    tag_fst.rmepsilon()
    #tag_fst.write("/tmp/tag_noeps.fst")
    tag_fst=fst.determinize(tag_fst)
    tag_fst.minimize()
   # tag_fst.write("/tmp/tag.fst")

    return tag_name_id, tag_fst



def get_new_grammar(tag_name_id:int, tag_fst:fst.VectorFst, g_fst:fst.VectorFst):

    replace_pairs = [(tag_name_id+9999,g_fst), (tag_name_id,tag_fst)]
    new_g_fst=fst.replace(replace_pairs, epsilon_on_replace=False)
    new_g_fst.rmepsilon().minimize()
    return new_g_fst

def expand_grammar(isym,g_fst,tag_file):
    print("Creating tag.fst")
    tag_name_id, tag_fst = create_tag_fst(isym, tag_file)
    print(tag_name_id)
    print("Inserting in G.fst")
    new_g_fst=get_new_grammar(tag_name_id,tag_fst,g_fst)
    return new_g_fst


if __name__=="__main__":
    if len(sys.argv)!=4:
        print("Usage: expand_grammmarfst.py <path_to_words_symboltable> <path to the grammar> <path to G.fst>")
    symbol_table_file=sys.argv[1]
    g_fst=fst.VectorFst().read(sys.argv[3])
    tag_file=sys.argv[2]

    isym=fst.SymbolTable.read_text(symbol_table_file)

    tag_name_id, tag_fst=create_tag_fst(isym, tag_file)
    new_g_fst=get_new_grammar(tag_name_id,tag_fst,g_fst)
    out_filename=sys.argv[3].replace('.fst', '_new.fst')
    new_g_fst.write(out_filename)
