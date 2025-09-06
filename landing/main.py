from fasthtml.common import *

# Add Tailwind CSS
tlink = Link(rel="stylesheet", href="https://cdn.tailwindcss.com")
app, rt = fast_app(hdrs=(tlink,))

@rt("/")
def get():
    return Html(
        Head(
            Title("Alternative Data for Investing - AltSignals"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Link(rel="stylesheet", href="https://cdn.tailwindcss.com"),
            Style("""
                .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
                .card-hover:hover { transform: translateY(-2px); transition: all 0.3s ease; }
                .text-gradient { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            """)
        ),
        Body(
            # Header
            Header(
                Nav(
                    Div(
                        # Logo
                        A(
                            Div(
                                Span("⚡", cls="text-2xl mr-2"),
                                Span("AltSignals", cls="text-xl font-bold text-gray-800"),
                                cls="flex items-center"
                            ),
                            href="/", 
                            cls="flex items-center"
                        ),
                        
                        # Desktop Navigation
                        Div(
                            A("Datasets", href="#", cls="text-gray-600 hover:text-gray-800 px-3 py-2 rounded-md text-sm font-medium"),
                            A("Top Stocks", href="#", cls="text-gray-600 hover:text-gray-800 px-3 py-2 rounded-md text-sm font-medium"),
                            A("Stock Alerts", href="#", cls="text-gray-600 hover:text-gray-800 px-3 py-2 rounded-md text-sm font-medium"),
                            A("AI Stock Picks", href="#", cls="text-gray-600 hover:text-gray-800 px-3 py-2 rounded-md text-sm font-medium"),
                            cls="hidden md:flex space-x-4"
                        ),
                        
                        # Auth buttons
                        Div(
                            A("Log in", href="#", cls="text-gray-600 hover:text-gray-800 px-3 py-2 rounded-md text-sm font-medium"),
                            A("Sign up", href="#", cls="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-md text-sm font-medium ml-2"),
                            cls="flex items-center"
                        ),
                        
                        cls="flex justify-between items-center"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="bg-white shadow-sm border-b"
            ),
            
            # Hero Section
            Section(
                Div(
                    Div(
                        # Left side - Content
                        Div(
                            H1("Make Better Investment Decisions With Better Data", cls="text-4xl md:text-5xl font-bold text-gray-900 mb-6 leading-tight"),
                            P("Get unique AI stock picks, stock alerts, and thousands of alternative insights. Complement your due diligence with AltSignals.", cls="text-xl text-gray-600 mb-8 leading-relaxed"),
                            
                            # Search bar
                            Div(
                                Input(
                                    placeholder="Search Stocks & Companies",
                                    cls="flex-1 px-4 py-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                                ),
                                Button(
                                    "🔍",
                                    cls="px-6 py-3 bg-green-500 hover:bg-green-600 text-white rounded-r-lg font-medium"
                                ),
                                cls="flex mb-6 max-w-md"
                            ),
                            
                            P(
                                "Or ",
                                A("Sign up", href="#", cls="text-red-500 font-medium hover:text-red-600"),
                                " (for free) to get started.",
                                cls="text-gray-600 mb-8"
                            ),
                            
                            # Popular links
                            Div(
                                Div(
                                    Span("Popular Stocks: ", cls="text-gray-600 mr-2"),
                                    A("Tesla", href="#", cls="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm mr-2 hover:bg-blue-200"),
                                    A("Apple", href="#", cls="bg-green-100 text-green-800 px-2 py-1 rounded text-sm mr-2 hover:bg-green-200"),
                                    A("Nvidia", href="#", cls="bg-purple-100 text-purple-800 px-2 py-1 rounded text-sm hover:bg-purple-200"),
                                    cls="mb-3"
                                ),
                                Div(
                                    Span("Trending Stocks: ", cls="text-gray-600 mr-2"),
                                    A("Reddit Stocks", href="#", cls="bg-orange-100 text-orange-800 px-2 py-1 rounded text-sm mr-2 hover:bg-orange-200"),
                                    A("Top Gainers", href="#", cls="bg-pink-100 text-pink-800 px-2 py-1 rounded text-sm mr-2 hover:bg-pink-200"),
                                    A("Penny Stocks", href="#", cls="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm hover:bg-yellow-200"),
                                    cls="mb-3"
                                ),
                                Div(
                                    Span("Popular Crypto: ", cls="text-gray-600 mr-2"),
                                    A("Bitcoin", href="#", cls="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm mr-2 hover:bg-yellow-200"),
                                    A("Ethereum", href="#", cls="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm hover:bg-blue-200"),
                                ),
                                cls="space-y-2"
                            ),
                            
                            cls="lg:w-1/2 lg:pr-8"
                        ),
                        
                        # Right side - Dashboard mockup
                        Div(
                            Div(
                                # Dashboard mockup
                                Img(
                                    src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23f8fafc'/%3E%3Crect x='20' y='20' width='360' height='40' fill='white' rx='8'/%3E%3Crect x='30' y='30' width='60' height='20' fill='%23e2e8f0' rx='4'/%3E%3Crect x='100' y='30' width='80' height='20' fill='%23e2e8f0' rx='4'/%3E%3Crect x='20' y='80' width='170' height='200' fill='white' rx='8'/%3E%3Cpath d='M40 200 L60 180 L80 160 L100 140 L120 120 L140 100 L160 80' stroke='%2310b981' stroke-width='3' fill='none'/%3E%3Crect x='210' y='80' width='170' height='90' fill='white' rx='8'/%3E%3Ccircle cx='295' cy='125' r='30' fill='%2306b6d4'/%3E%3Ctext x='295' y='130' text-anchor='middle' fill='white' font-size='16' font-weight='bold'%3E76%3C/text%3E%3Crect x='210' y='190' width='170' height='90' fill='white' rx='8'/%3E%3C/svg%3E",
                                    alt="AltSignals Dashboard",
                                    cls="w-full h-auto rounded-lg shadow-2xl"
                                ),
                                cls="relative"
                            ),
                            cls="lg:w-1/2 mt-8 lg:mt-0"
                        ),
                        
                        cls="flex flex-col lg:flex-row items-center"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="bg-gradient-to-br from-gray-50 to-white py-16 lg:py-24"
            ),
            
            # Data Insights Cards
            Section(
                Div(
                    # Sentiment cards row
                    Div(
                        Div(
                            Div(
                                Span("Sentiment", cls="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-1 rounded-full"),
                                H3("Lucid Motors (LCID)", cls="font-bold text-gray-900 mt-2"),
                                Div(
                                    Span("51", cls="text-2xl font-bold text-gray-900"),
                                    Span("9.8%", cls="text-green-500 font-medium ml-2"),
                                    cls="flex items-baseline mt-1"
                                ),
                                cls="p-4"
                            ),
                            cls="bg-white rounded-lg shadow-md border border-gray-200 card-hover"
                        ),
                        
                        Div(
                            Div(
                                Span("Reddit Mentions", cls="text-xs font-medium text-orange-600 bg-orange-100 px-2 py-1 rounded-full"),
                                H3("SPDR S&P 500 ETF (SPY)", cls="font-bold text-gray-900 mt-2"),
                                Div(
                                    Span("654", cls="text-2xl font-bold text-gray-900"),
                                    Span("142.5%", cls="text-green-500 font-medium ml-2"),
                                    cls="flex items-baseline mt-1"
                                ),
                                cls="p-4"
                            ),
                            cls="bg-white rounded-lg shadow-md border border-gray-200 card-hover"
                        ),
                        
                        Div(
                            Div(
                                Span("Job Posts", cls="text-xs font-medium text-green-600 bg-green-100 px-2 py-1 rounded-full"),
                                H3("Mondelez (MDLZ)", cls="font-bold text-gray-900 mt-2"),
                                Div(
                                    Span("1,599", cls="text-2xl font-bold text-gray-900"),
                                    Span("30.5%", cls="text-green-500 font-medium ml-2"),
                                    cls="flex items-baseline mt-1"
                                ),
                                cls="p-4"
                            ),
                            cls="bg-white rounded-lg shadow-md border border-gray-200 card-hover"
                        ),
                        
                        Div(
                            Div(
                                Span("Employee Business Outlook", cls="text-xs font-medium text-pink-600 bg-pink-100 px-2 py-1 rounded-full"),
                                H3("Cincinnati Financial (CINF)", cls="font-bold text-gray-900 mt-2"),
                                Div(
                                    Span("68", cls="text-2xl font-bold text-gray-900"),
                                    Span("12.8%", cls="text-green-500 font-medium ml-2"),
                                    cls="flex items-baseline mt-1"
                                ),
                                cls="p-4"
                            ),
                            cls="bg-white rounded-lg shadow-md border border-gray-200 card-hover"
                        ),
                        
                        cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="py-16 bg-gray-50"
            ),
            
            # Better Investments Section
            Section(
                Div(
                    Div(
                        # Left side - Content
                        Div(
                            H2("Better Investments with Better Data", cls="text-3xl md:text-4xl font-bold text-gray-900 mb-6"),
                            P("At AltSignals, we go beyond traditional financial data in our investment analysis, integrating a variety of alternative data points to provide a more comprehensive view. Our unique methodology encompasses insights such as job postings, website traffic, customer satisfaction, app downloads, and social media trends, in addition to our rigorous financial and technical insights and analysis. This multifaceted approach enables us to uncover unique investment opportunities and help our users make more informed investments.", cls="text-lg text-gray-600 mb-8 leading-relaxed"),
                            A("Sign up", href="#", cls="bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 rounded-lg font-medium text-lg"),
                            cls="lg:w-1/2 lg:pr-8"
                        ),
                        
                        # Right side - Dashboard image
                        Div(
                            Img(
                                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 500 400'%3E%3Crect width='500' height='400' fill='%23111827' rx='12'/%3E%3Crect x='20' y='20' width='460' height='50' fill='%231f2937' rx='8'/%3E%3Crect x='30' y='30' width='100' height='30' fill='%2310b981' rx='4'/%3E%3Crect x='140' y='30' width='80' height='30' fill='%23374151' rx='4'/%3E%3Crect x='20' y='90' width='460' height='280' fill='%231f2937' rx='8'/%3E%3Cpath d='M50 300 L100 250 L150 200 L200 180 L250 160 L300 140 L350 120 L400 100 L450 80' stroke='%2310b981' stroke-width='4' fill='none'/%3E%3Cpath d='M50 320 L450 320' stroke='%23374151' stroke-width='2'/%3E%3Cpath d='M50 80 L50 320' stroke='%23374151' stroke-width='2'/%3E%3Ccircle cx='400' cy='200' r='60' fill='%2306b6d4'/%3E%3Ctext x='400' y='210' text-anchor='middle' fill='white' font-size='24' font-weight='bold'%3E76%3C/text%3E%3C/svg%3E",
                                alt="AltSignals Analytics Dashboard",
                                cls="w-full h-auto rounded-lg shadow-xl"
                            ),
                            cls="lg:w-1/2 mt-8 lg:mt-0"
                        ),
                        
                        cls="flex flex-col lg:flex-row items-center"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="py-16 lg:py-24 bg-white"
            ),
            
            # Stats Section
            Section(
                Div(
                    Div(
                        Div(
                            H3("40k+", cls="text-4xl md:text-5xl font-bold text-white mb-2"),
                            P("registered members", cls="text-blue-200"),
                            cls="text-center"
                        ),
                        Div(
                            H3("100+", cls="text-4xl md:text-5xl font-bold text-white mb-2"),
                            P("unique daily stock alerts", cls="text-blue-200"),
                            cls="text-center"
                        ),
                        Div(
                            H3("High", cls="text-4xl md:text-5xl font-bold text-white mb-2"),
                            P("win-rate on AI Stock picks", cls="text-blue-200"),
                            cls="text-center"
                        ),
                        Div(
                            H3("100k+", cls="text-4xl md:text-5xl font-bold text-white mb-2"),
                            P("daily insights", cls="text-blue-200"),
                            cls="text-center"
                        ),
                        cls="grid grid-cols-2 lg:grid-cols-4 gap-8"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="bg-slate-800 py-16 lg:py-20"
            ),
            
            # Impactful Insights Section
            Section(
                Div(
                    H2("Impactful Insights and Alerts", cls="text-3xl md:text-4xl font-bold text-gray-900 mb-8 text-center"),
                    P("At AltSignals, we offer cutting-edge solutions that enables you to stay informed of any developments related to companies in your portfolio. By signing up with us, you'll get impactful stock alerts whenever a company experiences a surge in popularity on social media platforms such as Reddit, an uptick in webpage traffic, or a shift in employee satisfaction levels, among other critical indicators.", cls="text-lg text-gray-600 mb-8 text-center max-w-4xl mx-auto leading-relaxed"),
                    Div(
                        A("Sign up", href="#", cls="bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 rounded-lg font-medium text-lg"),
                        cls="text-center"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="py-16 lg:py-24 bg-gray-50"
            ),
            
            # AI Stock Picks Section
            Section(
                Div(
                    H2("Unique AI stock picks", cls="text-3xl md:text-4xl font-bold text-gray-900 mb-8 text-center"),
                    P("Unlock smarter investing with AltSignals' AI Stock Picks. Boasting a 80% win rate, our advanced algorithms sift through thousands of stocks daily, offering you top-notch buy or sell signals. We don't just rely on traditional financial data – we delve deeper. By harnessing the power of both traditional and alternative data – from social media trends, to employee sentiment, to user trends – we paint a fuller picture, enabling sharper, more informed investment decisions.", cls="text-lg text-gray-600 mb-6 text-center max-w-4xl mx-auto leading-relaxed"),
                    P("With AltSignals, you're not just keeping pace; you're staying ahead of the curve, armed with the insights that matter.", cls="text-lg text-gray-600 mb-8 text-center max-w-4xl mx-auto leading-relaxed"),
                    Div(
                        A("Sign up", href="#", cls="bg-orange-500 hover:bg-orange-600 text-white px-8 py-3 rounded-lg font-medium text-lg"),
                        cls="text-center"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="py-16 lg:py-24 bg-white"
            ),
            
            # Testimonial Section
            Section(
                Div(
                    H2("Don't take our word for it. Take theirs.", cls="text-3xl md:text-4xl font-bold text-gray-900 mb-12 text-center"),
                    Div(
                        Div(
                            P("As an investor who has just started this year, I've tried various Stock websites, but Altindex is the best alternative data provider. It offers invaluable insights, which I haven't seen other websites offer, and even allows me to see lobbying costs. Also, I have heard the AI-backed stock picking service has consistently delivered impressive results, and the stock portfolio alerts are great, so I see my portfolio performance relative to the market.", cls="text-lg text-gray-700 mb-6 italic leading-relaxed"),
                            Div(
                                P("Kenneth Thakur", cls="font-bold text-gray-900"),
                                P("Retail investor", cls="text-gray-600"),
                                cls="text-left"
                            ),
                            cls="p-8"
                        ),
                        cls="bg-white rounded-xl shadow-lg border border-gray-200 max-w-4xl mx-auto"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="py-16 lg:py-24 bg-gray-50"
            ),
            
            # Final CTA Section
            Section(
                Div(
                    H2("Made it all the way down here?", cls="text-3xl md:text-4xl font-bold text-gray-900 mb-4 text-center"),
                    P("Then it's time to try AltSignals and to make more informed investments.", cls="text-xl text-gray-600 mb-2 text-center"),
                    P("Every day we find thousands of insights and we bring everything to you in an easy-to-use dashboard.", cls="text-lg text-gray-600 mb-8 text-center"),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="py-16 lg:py-24 bg-white"
            ),
            
            # Footer
            Footer(
                Div(
                    Div(
                        P("© 2025 AltSignals. All rights reserved.", cls="text-gray-600 text-center"),
                        cls="border-t border-gray-200 pt-8"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="bg-gray-50 py-8"
            ),
            
            cls="min-h-screen bg-white"
        )
    )

if __name__ == '__main__':
    serve()

