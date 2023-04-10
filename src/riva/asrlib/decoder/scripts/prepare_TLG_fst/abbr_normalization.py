import sys


def abbr_map(abbr_mapfile, grammar_file):
    abbr_maps = {}
    with open(abbr_mapfile, 'r') as abbr_map_fp:
        for line in abbr_map_fp:
            (abbr, expansion) = line.strip().split('\t', maxsplit=1)
            abbr_map[abbr] = expansion
    if len(abbr_maps) > 0:
        with open(grammar_file, 'r') as grm_ifp, open(grammar_file + '.exp', 'w') as ofp:
            text = '\n'.join(grm_ifp.readlines())  # grammars are not expected to be large
            for abbr in abbr_maps:
                if abbr in text:
                    text.replace(abbr, abbr_maps[abbr])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: abbr_normalization.py <path_to_abbr_mapping file> <path to the grammar file>")

    # this should be a tsv of format <abbreviation>\t<spoken_expansion>
    # this will also be used in ITN to denorm the output
    abbr_mapfile = sys.argv[1]

    grammar_file = sys.argv[2]  # this is the file containing list of entities with abbreviations to be expanded
    abbr_map(abbr_mapfile, grammar_file)
