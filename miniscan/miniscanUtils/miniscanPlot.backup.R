args <- commandArgs(trailingOnly=T)
miniscan <- args[1] # miniscan output
prefix <- args[2]

# load data
d <- read.delim(miniscan, header=F)

# number of chromosomes/contigs
nchr <- length(unique(d[, 1]))

# plot

plot_height <- 8 * nchr / 10
pdf(paste0(prefix, ".miniscan.pdf"), width=5, height=plot_height)
par(mar=c(4, 1, 3, 1))
canvas_x_scale <- 1
canvas_y_scale <- 1
plot(NULL, NULL, xlab="", ylab="",
     xlim=c(0,canvas_x_scale),
     ylim=c(0,canvas_y_scale),
     bty="n",xaxt="n",yaxt="n")

plot_chr <- c("mini", paste0("chr", 1:7))

# minic
minic_max_round <- round(max(d[, 5]),1)
minic_label <- seq(0, minic_max_round, by=0.1)
minic_label_pos <- seq(1, 0, by=-1/(length(minic_label) - 1))
axis(1, at=minic_label_pos, labels=minic_label)

minic_x_converter <- function(x) {
# convert minic value to positions on x-axis
  (minic_max_round - x) / minic_max_round
}

# chr_position
gap <- 0.01
chr_ends <- tapply(d[, 3], d[, 1], max)/1000000
y_converter <- function(chrname, pos, gap, chr_ends, plot_chr) {
# convert chromosome positions to y-axis plotting coordinates
  chr_ends <- chr_ends[plot_chr]
  chr_accum <- sum(chr_ends)
  dist_per_unit <- (canvas_y_scale - (length(plot_chr) - 1) * gap) / chr_accum # distance per Mb
  chr_order_num <- which(plot_chr==chrname)
  if (chr_order_num > 1) {
    cur_accum <- sum(chr_ends[1:(chr_order_num-1)])
  } else {
    cur_accum <- 0
  }
  cur_coordinates <- (cur_accum + pos) * dist_per_unit + gap * (chr_order_num-1)
  cur_coordinates
}

base <- 0
for (chr in plot_chr) {
  # minic
  minic_val <- d[d[, 1] == chr, 4]
  gene_prop <- d[d[, 1] == chr, 5]
  # chr_position
  chr_pos <- (d[d[, 1] == chr, 2] + d[d[, 1] == chr, 3]) / 2000000
  chr_end_pos <- max(d[d[, 1] == chr, 3]) / 1000000
  chr_plot_coordinates <- y_converter(chrname=chr, pos=chr_pos, gap=gap, chr_ends=chr_ends, plot_chr=plot_chr)
  chr_end_coordinate <- y_converter(chrname=chr, pos=chr_end_pos, gap=gap, chr_ends=chr_ends, plot_chr=plot_chr)
  # plot
  lines(minic_x_converter(minic_val), chr_plot_coordinates, col="purple", lwd=1, xpt=T)
  lines(gene_x_converter(gene_prop), chr_plot_coordinates, col="gray90", lwd=1, xpt=T)
  gene_x_lowess <- lowess(gene_x_converter(gene_prop), f=0.1)$y
  gene_x_lowess[gene_x_lowess<1] <- 1
  lines(gene_x_lowess, chr_plot_coordinates, col="orange", lwd=2, xpt=T)
  if (chr != "chr7") {
    abline(h=chr_end_coordinate + c(0, gap), lty=2, col="gray")
  }
  # adjust baseline
  #base <- chr_end_coordinate  + gap
  text(0.15, chr_end_coordinate, labels=chr, pos=1, cex=1)
}

mtext(strain, side=3, line=1, cex=1.5)
mtext("minic proportion", side=1, line=2.2, cex=1.2, at=0.5)
mtext("gene proportion", side=1, line=2.2, cex=1.2, at=1.5)
abline(v=1, col="gray50", lwd=2)

dev.off()
