#!bin/bash
result_dir=$1
#result_dir=/mnt/data1/jiaohm/mapping_methods/Analysis_rCRS-hg38-DNA_V1
cd $result_dir
mkdir tmp
#mkdir ./fragment_study/gene_mis_10_insertsize
#rm -rf gene_mis_10_insertsize
#output_dir="./fragment_study/region_splited_sam_insertsize"
#filenames=$(ls -l ./fragment_study/region_splited_sam/*.non7sDNA.bam | awk '{if($5>1000) print$9}')
filenames=$(ls ./*/*mis.*.bam)
#filenames=$(ls)
#cat ${result_dir}/sample_name.txt | while read name
for i in ${filenames}
do
    name=${i##*/}
    sample=${name%%.*}
    #cd ${result_dir}/${name}
    echo ${name}
    echo ${sample}
    #file_name=${line}
    #i=${name}.mis.10.bam
    java -Xmx10g -Djava.io.tmpdir=`pwd`/tmp -jar /mnt/hdd/softs/picard/picard-tools-1.81/CollectInsertSizeMetrics.jar I=${i} O=./${sample}/${name}.insertsize.txt H=./${sample}/${name}.insertsize.pdf VALIDATION_STRINGENCY=LENIENT 2>>./insertsize.analysis.log
done
rm -rf tmp
