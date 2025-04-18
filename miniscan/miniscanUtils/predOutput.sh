#!/bin/bash
echo -e "chr\tstart\tend\ttotal\tmini_percent" > $table_output
cat $pred_dir/*final.csv | grep "^data" -v \
	| sed 's/[,:-]/\t/g' | sed 's/.gz//g' \
	| cut -f 1,2,3,4,7 | sort -k1,1 -k2n,2 \
	>> $table_output

# plot
module load R
Rscript $rplot_script \
	$table_output \
	${fasta}.fai \
	$outdir \
	$prefix 1

# cleanup if specified
if [ $cleanup -eq 1 ]; then
	rm -rf $log_dir
	rm -rf $pred_dir
	rm -rf $indata_dir
fi
