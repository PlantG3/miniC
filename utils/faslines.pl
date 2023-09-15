#!/usr/bin/perl -w
# File: faslines.pl
# Author: Sanzhen Liu
# Date: 3/23/2023 

use strict;
use warnings;
use Getopt::Long;

my $minlen=0;
my $linelen = 99;
my $outdir = ".";
my $prefix = "";
my $help;
my $result = &GetOptions("minlen|i=i" => \$minlen,
                         "outdir|o=s" => \$outdir,
					 	 "prefix|p=s" => \$prefix,
						 "linelen|l=i"=> \$linelen,
						 "help|h" => \$help
);

# print help information if errors occur:
if ($help or !@ARGV) {
	&errINF;
	exit;
}

if (!-d $outdir) {
	`mkdir $outdir`;
}


my ($seqname, $outseqname);
my $seq = "";
open(IN, $ARGV[0]) || die;
while (<IN>) {
	chomp;
	if (/^>(\S+)/) {
		if (defined $seqname) {
			if (length($seq) >= $minlen) {
				$outseqname = $outdir."/".$prefix.$seqname;
				open(OUT, ">", $outseqname) ||  die;
				while (my $chunk = substr($seq, 0, $linelen, "")) {
					print OUT "$chunk\n";
				}
				close OUT;
			}
		}
		$seqname = $1;
		$seq = '';
	} else {
		$seq .= uc($_);
	}
}
# last sequence
if (length($seq) >= $minlen) {
	$outseqname = $outdir."/".$prefix.$seqname;
	open(OUT, ">", $outseqname) ||  die;
	while (my $chunk = substr($seq, 0, $linelen, "")) {
		print OUT "$chunk\n";
	}
	close OUT;
}
close IN;

sub errINF {
	print <<EOF;
Usage: perl $0 <input> [Options]
	Options
	--minlen|m <num> : minimum length of a sequence for an output (0)
	--outdir|o <str> : output directory ($outdir)
	--prefix|p <str> : file prefix name ($prefix)
	--linelen|l <num>: bp of each line ($linelen)
	--help|h         : help information
EOF
	exit;
}

