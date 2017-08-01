import xml.etree.ElementTree as ET 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('xml')
parser.add_argument('--mice','-m')
parser.add_argument('--tag','-t')
parser.add_argument('-o')
args = parser.parse_args()

if args.mice:
	mice = args.mice.split()
	tag = None
elif args.tag:
	tag = args.tag
	mice = []
else:
	raise NameError('Select mice to analyze (-m "INDIVIDUAL MICE" or -t PREFIX)')

tree = ET.parse(args.xml)
root = tree.getroot()
new_root = ET.Element('BehaviorExperiments')

for mouse in root.findall('mouse'):
	if mouse.get('mouseID') in mice or tag in mouse.get('mouseID'):
	#	print(experiments)
		for expt in mouse.findall('experiment'):
			if expt.get('experimentType') != "hiddenRewards":
				mouse.remove(expt)
		new_root.append(mouse)

outtree = ET.ElementTree(element=new_root)
outtree.write(args.o)