################################################
### Sanzhen Liu
### 1/27/2024
################################################
args <- commandArgs(trailingOnly=T)
miniscan_file <- args[1] # miniscan output
chrlen_file <- args[2] # chromosome lengths (2 columns)
outdir <- args[3] # output directory
prefix <- args[4]
if (length(args) > 4) {
	ymax <- args[5] # the maximum value for y-axis
	ymax <- as.numeric(ymax)
} else {
	ymax <- NULL
}

# load data
d <- read.delim(miniscan_file)
plot_chr <- d[, 1]
plot_pos <- round((d[, 2] + d[, 3]) / 2)
plot_perc <- d[, 5]

chrlen <- read.delim(chrlen_file, header=F)
colnames(chrlen) <- c("chr", "length")

# number of chromosomes/contigs
plotchr <- unique(d[, 1])
nchr <- length(plotchr)
chrlen <- chrlen[chrlen$chr %in% plotchr, ]

# plot
plot_width <- 7.5 * max(round(nchr / 10, 2), 1)
pdf(paste0(outdir, "/", prefix, ".miniscan.pdf"), width=plot_width, height=5)

# x-coordinates 
gap <- sum(chrlen$length)/100
if (is.null(ymax)) {
	ymax <- max(plot_perc)
}

# plot
plot(NULL, NULL, ylim=c(0, ymax),
     xlim=c(0, gap * (nchr - 1) + sum(chrlen$length)),
     xaxt="n", xlab="",
	   ylab="MiniC proportion",
     bty="n", main=prefix)

accum <- 0 
all.accum <- NULL
all.col <- NULL
all.chr <- NULL
centers <- NULL
for (i in 1:nchr) {
  all.accum <- c(all.accum, accum)
  pre.accum <- accum
  chr <- chrlen[i, "chr"]
  len <- chrlen[i, "length"]
  if (i/2 == round(i/2)) {
    plot_col = 'steelblue'
  } else {
    plot_col = "olivedrab"
  }
  ppos <- plot_pos[plot_chr == chr]
  pperc <- plot_perc[plot_chr == chr]
  lines(c(accum, accum+len), c(-(ymax/50), -(ymax/50)),
        col=plot_col, lwd=1, lend=1)
    
  lines(accum + ppos, pperc, cex=0.6, col=plot_col)
  accum <- accum + len + gap
  center.point <- (pre.accum + accum - gap)/2
  all.col <- c(all.col, plot_col)
  all.chr <- c(all.chr, chr)
  centers <- c(centers, center.point)
}

# x-axis
if (mean(nchar(all.chr)) >= 5) {
  #axis(side=1, at=centers, labels=all.chr, tick=F, las=2, cex.axis=0.6)
  text(x=centers, y=-0.06, adj=1, las=2, labels = all.chr, cex=0.8, srt=30, xpd=T)
} else {
  axis(side=1, at=centers, labels=all.chr, tick=F, cex.axis=0.8)
}

dev.off()

