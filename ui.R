### Libraries----------------------------------------------------------------------------------
library(shinydashboard)
library(shiny)
library(knitr)
library(rmarkdown)

rmdfiles <- c("outcomes.Rmd")
sapply(rmdfiles, knit, quiet = T)


### UI ----------------------------------------------------------------------------------------

ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = paste("IATD",format(Sys.time(), "%b %d"), sep = " "),
                  dropdownMenu(type = "notifications",
                               notificationItem(
                                 text = "Press F11 to fullscreen!",
                                 icon = icon("exclamation-triangle"),
                                 status = "warning"
                               )),
                  # dropdownMenu(type = "tasks", badgeStatus = "success",
                  #              taskItem(value = 95, color = "green",
                  #                       "Machine Learning Project"
                  #              ),
                  #              taskItem(value = 80, color = "aqua",
                  #                       "Visualizations"
                  #              ),
                  #              taskItem(value = 75, color = "yellow",
                  #                       "Automation"
                  #              )
                  #),
                  tags$li(a(href = 'https://www150.statcan.gc.ca/n1/dai-quo/index-eng.htm?HPA=1',
                            icon("power-off"),
                            title = "Back to The Daily"),
                          class = "dropdown"),
                  tags$li(a(href = 'https://www.statcan.gc.ca/eng/start',
                            img(src = 'logo.png',
                                title = "StatCan Home", height = "30px"),
                            style = "padding-top:10px; padding-bottom:10px;"),
                          class = "dropdown")
                  

                  
  ),
  dashboardSidebar(
    sidebarMenu(id='sidebar',
                menuItem("Main", tabName = "main", icon = icon("navicon")),
                menuItem("Motivation", tabName = "motivation", icon = icon("slideshare")),
                menuItem("Dataset", tabName = "dataset", icon = icon("database")),
                menuItem("NLP", tabName = "nlp", icon = icon("language")),
                menuItem("XGBoost", tabName = "xgboost", icon = icon("xing")),
                #menuItem("The Method", tabName = "method", icon = icon("code")),
                menuItem("FAQ & References", tabName = "faq", icon = icon("question"))
                
    )
    
  ),
  dashboardBody(
    
    tags$head(
      tags$link(rel="stylesheet", type="text/css", href="custom.css")
    ),
    
    tabItems(
      
      # 1st tab content------------------------------------------------------------------------
      tabItem(tabName = "main",
              
              fluidRow(
                tags$style(".topimg {
                           margin-top:-15px;
                           }"),
                div(class="topimg",img(src="template.png", height="100%", width="100%", align="center"))
                #div(class="topimg",img(src="template.png", height="650px", width="1150px", align="center"))
                #box(width = 12, height = 650, solidHeader = TRUE, img(src="template.png", width="1150px", height="650px", align="center"))
              )
      ),

      # 2nd tab content------------------------------------------------------------------------
      tabItem(tabName = "motivation",

              
              fluidRow(
                column(width = 9,
                       h2(style = "font-size: 300%;text-align:center;font-weight: bold;margin-top:0px;margin-left:160px;", "Motivation"),
                       
                       h3(style ="text-decoration: underline;font-size: 180%;text-align:justify", "Existing challenge"),
                       br(),
                       tags$ul(
                         tags$li(style= "font-size: 180%;;text-align:justify", "Every month, the analysts try to.."),
                         br(),
                         tags$li(style= "font-size: 180%;;text-align:justify", "Time consuming process")
                       ),
                       #br(),
                       h3(style ="text-decoration: underline;font-size: 180%;text-align:justify", "Proposed solution"),
                       br(),
                       tags$ul(
                         tags$li(style= "font-size: 180%;;text-align:justify", "A supervised machine learning approach to facilitate the export trade analysis at the company level"),
                         br(),
                         tags$li(style= "font-size: 180%;;text-align:justify", "The model will 'learn' all the..."), 
                         br(),
                         tags$li(style= "font-size: 180%;;text-align:justify", "It will try to predict"),
                         br(),
                         tags$li(style= "font-size: 180%;;text-align:justify", "Solution aims to give..")
                       )
                       #img(src="ptable1.png", height="200px", width="700px", align="center")
                ),
                column(width = 3,
                       br(),
                       br(),
                       br(),
                       br(),
                       br(),
                       tags$style(".frus {
                           margin-left:15px;
                                  }"),
                       tags$style(".happ {
                           margin-left:35px;
                                  }"),
                       img(class = "frus",src="frustrated1.png", width="235px", height="250px", align="center"),
                       br(),br(),br(),br(),br(),
                       img(class = "happ", src="happy.png", width="180x", height="250px", align="center")
                       
                )
                
              )
              

              
      ),
      # 3rd tab content------------------------------------------------------------------------
      tabItem(tabName = "dataset",
              h2(style = "font-size: 300%;text-align:center;font-weight: bold;margin-top:0px;", "Dataset"),
              fluidRow(column(width = 6,
              br(),
              #h2(style = "font-size: 300%;text-align:center;font-weight: bold;", "Dataset"),
              tags$ul(
                tags$li(style ="font-size: 190%;text-align:justify", "...data as a csv file"),
                br(),
                tags$li(style ="font-size: 190%;text-align:justify", "Contains information about each business, for instance"),
                br(),
                tags$li(style ="font-size: 190%;text-align:justify", "Variables change monthly for each business"),
                br(),
                tags$li(style ="font-size: 190%;text-align:justify", "Large dataset: ~50k entries")
                
                
                
                
              )
              
              ),
              column(width = 6,
                     
                     
                     img(src="timmies.png", width="590px", height="400px", align="center")
              ),
              fluidRow(
                column(11,
                       tags$div(
                         style="margin-left:50px;font-size: 130%;",
                         includeMarkdown("outcomes.md")
                       )
                       ),
                column(1)
              )
              
              )
              
      ),
      # 4th tab content------------------------------------------------------------------------
      tabItem(tabName = "nlp",
              fluidRow(
                column(width = 12,
                       h1(style="text-align:center; font-weight: bold; font-size: 300%;margin-top:0px;", "Natural Language Processing"),
                       #br(),
                       tabsetPanel(
                         tabPanel(title="The Challenge",
                                  fluidRow(
                                    column(width = 8,
                                           br(),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "The challenge with text input"),
                                           br(),
                                           tags$ul(
                                             tags$li(style= "font-size: 190%;;text-align:justify", "Machine learning algorithms prefer well defined fixed-length inputs and outputs."),
                                             br(),
                                             pre(style= "font-size: 130%;", "Unfixed lenght input: [1, 2, 3], [1, 2, 3, 4], [1, 2], etc"),
                                             
                                             br(),
                                             
                                             tags$li(style= "font-size: 190%;;text-align:justify", "The algorithms cannot work with raw text directly; the text must be converted into numbers."), 
                                             br(),
                                             code(style= "font-size: 100%;","Example of raw text:"),
                                             pre(style= "font-size: 130%;","WalmartCanadaVancouver"),
                                             br(),
                                             tags$li(style= "font-size: 190%;;text-align:justify", "The bag-of-words model will help us feature encode each.")
                                           )
                                    ),
                                    column(width = 4,
                                           tags$style(".mathy {
                                                      margin-left:5px;
                                                      }"),
                                           img(class = "mathy", src="math.png", width="350px", height="590px", align="right")
                                           )
                                  )
                         ),
                         tabPanel(title="The Model",
                                  fluidRow(
                                    column(width = 6,
                                           br(),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "The bag-of-words model"),
                                           br(),
                                           tags$ul(
                                             tags$li(style= "font-size: 190%;;text-align:justify", "The bag-of-words model is a method of extracting features from text for analytical use."),
                                             br(),br(),
                                             tags$li(style= "font-size: 190%;;text-align:justify", "Any information about the order, structure or content of words is ignored."), 
                                             br(),br(),
                                             tags$li(style= "font-size: 190%;;text-align:justify", "The model is only concerned with the occurance of known words in the document. The location in the document does not matter.")
                                           )
                                    ),
                                    column(width = 6,
                                           br(),
                                           br(),
                                           br(),
                                           br(),
                                           tags$style(".conf {
                                                      margin-left:25px;
                                                      }"),
                                           img(class = "conf", src="confused.png", width="515px", height="400px", align="center")
                                           )
                                    
                                  )
                         ),
                         tabPanel(title="N-grams",
                                  fluidRow(
                                    column(width = 8,
                                           br(),br(),
                                           p(style= "font-size: 190% ;text-align:justify", "N-grams are all combinations of adjacent words or letters of length", em("n") ,"that you
                                             can find in your source text. There are two types of N-grams, word and character. For example: "),
                                           br(),
                                           tags$ul(
                                             tags$li(style ="font-size: 190%;text-align:justify", "Given the word ",em("'fox'"),", all character 'bigrams' are:",
                                                     p(),
                                                     pre("'fo'\n'ox'")),
                                             br(),
                                             
                                             
                                             br(),
                                             
                                             tags$li(style ="font-size: 190%;text-align:justify", "Given the phrase ",em("'This is a fox'"),", all word 'bigrams' are:",
                                                     p(),
                                                     pre("'This is'\n'is a'\n'a fox'")), 
                                             br(),
                                             
                                             br()
                                           )
                                           ),
                                    column(width = 4,
                                           br(),br(),br(),br(),br(),
                                           img(src="fox.png", width="360px", height="400px", align="center")
                                           
                                           
                                           )
                                  )
                       ),
                         tabPanel(title="An Example",
                                  fluidRow(
                                    column(width = 8,
                                           br(),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Step 1: Collect the data "),
                                           #br(),
                                           h4(style= "font-size: 170%;", "Below is an example ...:"),
                                           pre("[1] AIR CANADA \n[2] AIR CANADA TERMINAL\n[3] AIR CANADA CARGO\n[4] AIR CANADA PURCHASING & CO\n[5] AAR AIR CANADA"),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Step 2: Obtain the vocabulary"),
                                           h4(style= "font-size: 170%;", "The unique words are (ignoring case and punctuation):"),
                                           pre('[1] "AIR"\n[2] "CANADA"\n[3] "TERMINAL"\n[4] "CARGO"\n[5] "PURCHASING"\n[6] "CO"\n[7] "AAR"'),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Step 3: Create document vectors"),
                                           tags$ul(
                                             # tags$li(style= "font-size: 170%;", "The next step is to score the words in each document. The objective is to turn each document of text into a vector."),
                                             # br(),
                                             tags$li(style= "font-size: 170%;", "Each textual term will be turned into a vector. This is done by scoring each word in the document."),
                                             br(),
                                             tags$li(style= "font-size: 170%;", "The presence of words are marked as a Boolean value: 0 for absent, 1 for present.") 
                                           ),
                                           br(),
                                           h4(style= "font-size: 170%;", 'The first entry “Air Canada” would look as follows:'),
                                           pre('[1] "AIR" = 1\n[2] "CANADA" = 1\n[3] "TERMINAL" = 0\n[4] "CARGO" = 0\n[5] "PURCHASING" = 0\n[6] "CO" = 0\n[7] "AAR" = 0'),
                                           br(),
                                           h4(style= "font-size: 170%;", 'The binary vector for “AIR CANADA”:'),
                                           pre('[1] "AIR CANADA" =  [1, 1, 0, 0, 0, 0, 0]'),
                                           br(),
                                           h4(style= "font-size: 170%;", "The other 4:"),
                                           pre('[2] "AIR CANADA TERMINAL" =     [1, 1, 1, 0, 0, 0, 0]\n[3] "AIR CANADA CARGO" =    [1, 1, 0,  1, 0, 0, 0]\n[4] "AIR CANADA PURCHASING & CO" =     [1, 1, 0, 0, 1, 1, 0]\n[5] "AAR AIR CANADA" =  [1, 1, 0, 0, 0, 0, 1]')
                                           
                                           
                                    ),
                                    column(width = 4,
                                    br(),
                                    br(),br(),br(),
                                    img(src="female-teacher.png", width="360px", height="400px", align="center"), br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),
                                    img(src="teacher.png", width="260px", height="400px", align="center")
                                    )
                                  )
                         )
                         
                         
                         
                         
                       )
                )
                
                
              )

      ),
      # 5th tab content------------------------------------------------------------------------
      tabItem(tabName = "xgboost",
              fluidPage(
                column(width = 12,
                       h1(style="text-align:center; font-weight: bold; font-size: 300%;margin-top:0px;", "Extreme Gradient Boosting"),
                       br(),
                       tabsetPanel(
                         tabPanel(title="N-grams",
                                  fluidRow(
                                    column(width = 8,
                                           br(),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Character N-grams"),
                                           br(),
                                           tags$ul(
                                             tags$li(style= "font-size: 170%;;text-align:justify", "In the latest version of the program, character N-grams of length 3 to 6 are applied."),
                                             
                                             br(),
                                             
                                             tags$li(style= "font-size: 170%;;text-align:justify", "In the “AIR CANADA” example, character ngrams of 3 would be (ignoring case):"), 
                                             br(),
                                             pre('[1] "A_I_R"\n[2] "I_R_C"\n[3] "R_C_A"\n[4] "C_A_N"\n[5] "A_N_A"\n[6] "N_A_D"\n[7] "A_D_A"'),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "Chracter N-grams are empirically better for shorter strings, and they are more robust to spelling difference and typos. For instance:"),
                                             pre('[1] AIR CANADA\n[2] AIR CANDA'),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "“A_I_R”, “I_R_C”, “R_C_A” and “C_A_N” are still present in character N-grams. Word N-grams will take “CANADA” and “CANDA” as two distinct words.")
                                              
                                    )),
                                    column(width = 4,
                                           # tags$style(".rightimg {
                                           #            margin-left:81px;
                                           #            }"),
                                           tags$style(".air {
                                                      margin-left:25px;
                                           }"),
                                           br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),br(),
                                           img(class = "air", src="air-canada.png", width="300px", height="245px", align="left")
                                           #br(),
                                           #img(class = "rightimg1", src="rmarkdown.png", width="200px", height="245px", align="center")
                                           )
                                  )
                         ),
                         tabPanel(title="Files",
                                  fluidRow(
                                    column(width = 8,
                                           br(),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Files"),
                                           br(),
                                           p(style= "font-size: 170%;;text-align:justify", "The program contains 3 different Sub-programs: Random Search.Rmd, Train Model.Rmd and Prediction.Rmd"),
                                           p(style= "font-size: 170%;;text-align:justify", "All computations are done in the R environment"),
                                           br(),
                                           tags$ol(
                                             
                                             tags$li(style= "font-size: 170%;;text-align:justify", "Random Search.Rmd helps us find the best parameters based on some pre-determined range for the parameteres"),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "Train Model.Rmd uses the parameters we obtained from Random Search.Rmd to train the model. Training accuracy is provided for validation purpose"),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "Prediction.Rmd uses the model to predict the BNs. The output is an input * nclass table that shows the predicted probability of each input belonging to each class"),
                                             br()
                                             
                                           )
                                           
                                    ),
                                    column(width = 4,
                                           tags$style(".rightimg {
                                                      margin-left:81px;
                                                      }"),
                                           tags$style(".rightimg1 {
                                                      margin-left:95px;
                                                      }"),
                                           br(),br(),br(),br(),
                                           img(class = "rightimg", src="rstudio.png", width="230px", height="245px", align="center"),
                                           br(),
                                           img(class = "rightimg1", src="rmarkdown.png", width="200px", height="245px", align="center")
                                           )
                                    
                                  )
                         ),
                         tabPanel(title="Overview",
                                  fluidRow(
                                    column(width = 8,
                                           br(),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Overview"),
                                           br(),
                                           tags$ul(
                                             tags$li(style= "font-size: 170%;;text-align:justify", "Instead of a tree based model, a linear model is used to predict the output."),
                                             br(),
                                             withMathJax(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "The XGBoost parameter ‘multi:softprob’ is used to calculate the predicted probability of each data point belonging to each class. L = (formula)"),
                                             #div("more math here $$\\sqrt{2}$$"),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "L2 regularization is applied to avoid over-fitting. It adds the sum of the square of the weights to the loss funjction."),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "An ‘unknown’ class is created using the other ."),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "In the lastest version, the learning rate ‘eta’ = , and regularization term ‘lambda’ ="),
                                             br())
                                    ),
                                    column(width = 4)
                                    
                                  )
                         ),
                         tabPanel(title="Random Search",
                                  fluidRow(
                                    column(width = 8,
                                           br(),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Random Search"),
                                           br(),
                                           tags$ul(
                                             tags$li(style= "font-size: 170%;;text-align:justify", "80% as training set, 20% as testing set."),
                                             br(),
                                             pre('nrow(train)\n[1] 7483'),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "Duplicate rows are deleted. Punctuations and spaces are removed ."),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "eta_range = [5e-05, 5e-02], lambda_range = [1e-06, 1e-02] and 5-folds cross validation is applied."),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "In the begining of each iteration, the program will randomly pick a set of pamameters and train the model for 50 rounds. Evaluation metric ‘Log-loss’ is used to compare the performance."),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "The set of parameters that scores the lowest ‘log-loss’ will be saved as ‘best_param’."),
                                             br(),
                                             tags$li(style= "font-size: 170%;;text-align:justify", "The testing accuracy for the 'best_param is -%'"),
                                             br()
                                             )
                                    ),
                                    column(width = 4)
                                    
                                  )
                         ),
                         tabPanel(title="Train Model",
                                  fluidRow(
                                    column(width = 10,
                                           br(),
                                           h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Train Model"),
                                           br(),
                                           p(style= "font-size: 160% ;text-align:justify", "Vocabulary for are created and saved as Rdata files. All ordering of the words is
                                             discarded and we have a consistent way of extracting features from any, ready for use in modelling"),
                                           br(),
                                           tags$ul(
                                             tags$li(style ="font-size: 160%;text-align:justify", "A sparse matrix for the vocabulary and created for training. The 301 classes are encoded as numeric numbers from 0 - 300.",
                                                     p(),
                                                     pre("dtrain <- xgb.DMatrix(dtm_train , label =  as.integer(ytrain) - 1)")),
                                             br(),
                                             
                                             
                                             br(),
                                             
                                             tags$li(style ="font-size: 160%;text-align:justify", "The model is trained for 100 rounds and it is using the whole data set and ‘best-param’.",
                                                     p(),
                                                     pre("model <- xgb.train(data=dtrain, params = best_param ,nrounds = 100)")), 
                                             br(),
                                             tags$li(style ="font-size: 160%;text-align:justify", "The model is saved for prediction.",
                                                     p(),
                                                     pre("xgb.save(model, 'model.model')")), 
                                             br()
                                           )
                                           )
                                    #column(width = 4)
                                  )
                       ),
                       tabPanel(title="Prediction",
                                fluidRow(
                                  column(width = 10,
                                         br(),
                                         h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Prediction"),
                                         br(),
                                         tags$ul(
                                           tags$li(style= "font-size: 170%;", "The input data set is the 2018."),
                                           br(),
                                           tags$li(style= "font-size: 170%;", "“Score” the vocabulary for the test set using the vocabulary file we previously created. New documents contain words/strings outside of the vocabulary, can still be encoded, where only the occurences of known strings are scored and unkown strings are ignored."),
                                           br(),
                                           tags$li(style= "font-size: 170%;", "Using the XGBoost model to predict which these data points belong to."),
                                           br(),
                                           pre("pm1 <- predict(model, dtest, reshape = T)"),
                                           br(),
                                           tags$li(style= "font-size: 170%;", "The output CSV only shows the highest probability for each data points. Thresholds can be applied for the output csv file."),
                                           br(),
                                           pre('dd4[dd4 != do.call(pmax, dd4)[row(dd4)]] <- " "\n\ndd4 <- nwrite.csv(dd4, "highest prob.csv")')
                                           
                                           
                                                )
                                  ),
                                  column(width = 2)
                                  
                                )
                       ),
                       tabPanel(title="Results",
                                fluidRow(
                                  column(width = 8,
                                         br(),
                                         h2(style="text-align:left; font-weight: bold; font-size: 240%;", "Results"),
                                         br(),
                                         tags$ul(
                                           tags$li(style= "font-size: 170%;", "We are confident that our machine has strong pred...."),
                                           br(),
                                           tags$li(style= "font-size: 170%;", "Any entry that falls under the unknown class or has a very low probability (<5%) belonging to any of the classes."),
                                           br(),
                                           tags$li(style= "font-size: 170%;", "Updating the model periodically with matched data is recommended to maintain/increase the predictive accuracy."),
                                           br(),
                                           tags$li(style= "font-size: 170%;", "Testing accuracy on the ...")
                                           
                                         )
                                  ),
                                  column(width = 4,
                                         br(),br(),br(),br(),br(),br(),br(),br(),
                                         tags$style(".abc {
                                                      margin-left:50px;
                                                    }"),
                                         img(class = "abc", src="checklist.png", width="200px", height="245px", align="center")
                                         )
                                  
                                )
                       )
                       
                       
                       
                       
                )
                )

              )
              
      ),
      # 6th tab content------------------------------------------------------------------------
      tabItem(tabName = "method"
              
      ),
      # 7th tab content------------------------------------------------------------------------
      tabItem(tabName = "faq",
              fluidRow(
                column(width = 12,
                       tabsetPanel(
                         tabPanel(title="FAQ",
                                  fluidRow(
                                    column(width=12,
                                           h3(style="text-align:center", "Frequently Asked Questions"),
                                           br(),
                                           box(width = 12, solidHeader = TRUE, 
                                               
                                               h4(strong("Why is an n-gram range of 2-10 being used?")),
                                               p("This is the answer, it does not make sense. The main eason is because 2% of the initial follow up increase drastically.
                                                 When you think about it like that, the change in space accounts for more than 200 different user cases. In the end,
                                                 we decided this would make the most sense.")
                                               
                                               ),
                                           box(width = 12, solidHeader = TRUE, 
                                               
                                               h4(strong("Why did we use a linear booster instead of a tree booster?")),
                                               p("This is the answer, it does not make sense."),
                                               #pre("This text is preformatted."),
                                               #code("This text will be displayed as computer code."),
                                               tags$ul(
                                                 tags$li("Support Vector machine"),
                                                 tags$li("Decision Tree"), 
                                                 tags$li("Naive Bayes"), 
                                                 tags$li("Random Forest"),
                                                 tags$li("AdaBoost"),
                                                 tags$li("XGBoost")
                                               ),
                                               p("After considering all of the above algorithms, we did this and that and came up with the best of ideas.")
                                               
                                           ),
                                           box(width = 12, solidHeader = TRUE, 
                                               
                                               h4(strong("Why did we pick a threshold of 0.6 for the probability tables?")),
                                               p("This is the answer, it does not make sense. The main eason is because 2% of the initial follow up increase drastically.
                                                 When you think about it like that.")
                                               
                                               )
                                           )
                                           )
                                  ),
                         tabPanel(title="References",
                                  fluidRow(
                                    column(width=12,
                                           h3(style="text-align:left", "Helpful links"),
                                           br(),
                                           box(width = 12, solidHeader = TRUE, 
                                               tags$ol(
                                                 tags$li(a(href = 'https://medium.com/data-design/understanding-a-bit-xgboosts-generalized-linear-model-gblinear-bc1354187dfe',
                                                           "Linear boosters")),
                                                 tags$li(a(href = "https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html", 'An introduction XGBoost')), 
                                                 tags$li(a(href = "https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html", 'More on N-grams')), 
                                                 tags$li(a(href = "https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html", 'Text classification')), 
                                                 tags$li(a(href = "https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html", 'XGBoost parameters - Explained!')), 
                                                 tags$li(a(href = "https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html", 'More on the softmax function')),
                                                 tags$li(a(href = "https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html", 'Linear vs tree boosters'))
                                               )
                                           ),
                                           h3(style="text-align:left", "People who deserve credit"),
                                           br(),
                                           box(width = 12, solidHeader = TRUE,
                                               p("Stan for his help with writing the XGBoost model in R, as well as taking time out to test our program."),
                                               tags$ul(
                                                 tags$li("stan.hatko@canada.ca"))
                                           ),
                                           h3(style="text-align:left", "R packages used"),
                                           br(),
                                           box(width = 12, solidHeader = TRUE,
                                               
                                               tags$ul(
                                                 tags$li("xgboost"),
                                                 tags$li("dplyr"),
                                                 tags$li("rio"),
                                                 tags$li("rmarkdown")
                                               )
                                           )
                                           
                                           
                                           
                                    )
                                    
                                  )
                         )
                       )
                       )
              )
              

              )
              

              
      )
      

              )
  )
  

