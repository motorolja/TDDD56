# Copyright 2015 Nicolas Melot
#
# This file is part of Freja.
#
# Freja is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Freja is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Freja. If not, see <http://www.gnu.org/licenses/>.
#

append_combinations = function(data.frame, comb = combination)
{
  ## Append keys for column combination
  for(col in names(comb))
  {
    paste_str = paste(sep = "", "data.frame[,\"", col, "\"] = paste(sep = \".\"")
    for(src in as.character(comb[[col]]))
    {
      paste_str = paste(sep = "", paste_str, ", data.frame[,\"", src, "\"]")
    }
    paste_str = paste(sep = "", paste_str, ")")
  
    eval(parse(text = paste_str))
    data.frame[,col] <- as.factor(data.frame[,col])
  }

  return(data.frame)
}

apply_labels = function(data.frame, labels.list = labels, comb = combination)
{
  data.frame = append_combinations(data.frame, comb)

  ## Change values with their human-friendly aliases  
  for(col in names(labels.list)[names(labels.list) != "columns"])
  {
    ## Only apply the transformation if the column exists, otherwise skip the column
    if(col %in% names(data.frame))
    {
      ## If the corresponding column in the data frame is not a factor, then make it one
      if(!is.factor(data.frame[,col]))
      {
        data.frame[,col] <- as.factor(data.frame[,col])
      }
      
      ## Get levels and respective labels as from the label list
      levels = names(labels.list[[col]])
      labels = as.character(labels.list[[col]])
      
      ## Pad missing labels with values as they appear in data
      if(!all(levels(data.frame[,col]) %in% names(labels.list[[col]])))
      {
        ## Levels of data.frame, where levels that appear in labels appear first and ordered as in the list
        levels = unique(c(names(labels.list[[col]]), levels(data.frame[,col])))
        
        ## Concatenate labels with Levels of data.frame that are not in the list of labels
        labels = c(as.character(labels.list[[col]]), levels(factor(data.frame[,col][!data.frame[,col] %in% names(labels.list[[col]])])))
      }

      ## Reorder levels of factors in data.frame, and give them labels
      data.frame[,col] <- factor(data.frame[,col], levels=levels, labels=labels)
      
      ## Only keep factors actually used in the table
      data.frame[,col] <- factor(data.frame[,col])
    }
  }
  
  return(data.frame)
}

label = function(col, columns="columns", labels.list = labels)
{
  if(col %in% names(labels.list[[columns]]))
    return(labels.list[[columns]][col])
  else
    return(col)
}

freja_colors = function(data.frame, col, colors.list = colors, default = c("#BB0000", "#66BB00", "#0000BB"), comb = combination, sep = "_", inter=".")
{
  ## Generate a vector of default colors
  def = colorRampPalette(default)(abs(length(unique(data.frame[,col])) - length(colors[[paste(col, collapse = sep)]])))
  defindex=1
  ## Get the list of colors we are interested in
  colors = colors.list[[paste(col, collapse = sep)]]
 
  ## Keep only unique occurences of rows grouped by col 
  data.frame = unique(data.frame[,col])
  ## Add a new column in data frame out of columns whose combined value gets one color
  data.frame$color = apply(data.frame[,col] , 1, function(x) paste(trimws(x), collapse = "."))
  ## Turn new coloumn to the corresponding color
  data.frame$color = unlist(lapply(data.frame[,"color"], function(x) {if( x %in% names(colors)) { return (colors[[x]])} else { col = def[[defindex]]; defindex <<- defindex + 1; return(col)}}))
  ## Replace content by reader-friendly content
  data.frame = apply_labels(data.frame)
  ## Associate reader-friendly content to corresponding color
  data.frame$name = apply(data.frame[,col] , 1, function(x) paste(trimws(x), collapse = inter))
  ## Keep only name and color
  data.frame = data.frame[,c("name", "color")]
  ## Form a list of colors from data frame
  col = data.frame$color
  ## Index colors by their associated name
  names(col) = data.frame$name

  return(col)
}

combination = list(
	)

labels = list(
	nb_threads = c("seq" = "Serial"),
	pattern = c("uniform-random" = "Random (uniform)", "increasing" = "Increasing", "decreasing" = "Decreasing", "constant" = "Constant"),
	implementation = c("drake" = "Drake", "parallel" = "Parallel"),
	columns = c("time" = "Time")
)

colors = list(
	)

