from argparse import ArgumentParser, Namespace

from tensors_analyzer import TensorsAnalyzer

def main(args: Namespace) -> None:
    tensors_analyzer = TensorsAnalyzer(args.filename_constraint, verbose=True)
    tensors_analyzer.load_tensors_by_multiple_folder(args.foldername_list)
    tensors_analyzer.draw_distribution(args.save_name, args.start_level, args.end_level)

if __name__ == "__main__":
    parser = ArgumentParser(description='Analyze tensors')
    parser.add_argument('--foldername_list', nargs='+', help='The list of folder names containing the tensors')
    parser.add_argument('--save_name', type=str, help='The name to save the distribution plot')
    parser.add_argument('--start_level', type=int, help='The start level of the histogram')
    parser.add_argument('--end_level', type=int, help='The end level of the histogram')
    parser.add_argument('--verbose', action='store_true', help='Print debug messages')
    parser.add_argument('--filename_constraint', type=str, help='The constraint for the filename')
    args = parser.parse_args()
    main(args)
    