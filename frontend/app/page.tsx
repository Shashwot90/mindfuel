import type React from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { MoveRight, Brain, Moon, Sparkles } from "lucide-react"
import { HeroAnimation } from "@/components/hero-animation"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-sky-50 to-white dark:from-slate-950 dark:to-slate-900">
      <header className="container mx-auto py-6 px-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Brain className="h-8 w-8 text-teal-500" />
          <span className="text-2xl font-bold bg-gradient-to-r from-teal-500 to-sky-500 bg-clip-text text-transparent">
            MindFuel
          </span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/login">
            <Button variant="ghost" className="text-slate-700 dark:text-slate-200">
              Login
            </Button>
          </Link>
          <Link href="/register">
            <Button className="bg-gradient-to-r from-teal-500 to-sky-500 text-white">Sign Up</Button>
          </Link>
        </div>
      </header>

      <main>
        <section className="container mx-auto px-4 py-20 flex flex-col md:flex-row items-center gap-12">
          <div className="flex-1 space-y-6">
            <h1 className="text-4xl md:text-6xl font-bold text-slate-800 dark:text-white">
              Your AI-Powered{" "}
              <span className="bg-gradient-to-r from-teal-500 to-sky-500 bg-clip-text text-transparent">
                Mental Health
              </span>{" "}
              Companion
            </h1>
            <p className="text-lg text-slate-600 dark:text-slate-300 max-w-xl">
              MindFuel uses advanced AI to analyze your mood, recommend personalized meditation exercises, and provide
              sleep suggestions tailored to your needs.
            </p>
            <div className="flex flex-wrap gap-4 pt-4">
              <Link href="/register">
                <Button size="lg" className="bg-gradient-to-r from-teal-500 to-sky-500 text-white">
                  Get Started <MoveRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/about">
                <Button
                  size="lg"
                  variant="outline"
                  className="border-teal-500 text-teal-500 hover:text-teal-600 hover:bg-teal-50 dark:hover:bg-slate-800"
                >
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
          <div className="flex-1 relative h-[400px] w-full">
            <HeroAnimation />
          </div>
        </section>

        <section className="container mx-auto px-4 py-20">
          <h2 className="text-3xl font-bold text-center mb-16 text-slate-800 dark:text-white">
            How MindFuel Helps You
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Brain className="h-10 w-10 text-teal-500" />}
              title="Mood Analysis"
              description="Our AI analyzes your mood through text and voice inputs to provide personalized coping strategies."
            />
            <FeatureCard
              icon={<Sparkles className="h-10 w-10 text-sky-500" />}
              title="Guided Meditation"
              description="Access AI-generated guided meditations and mindfulness exercises tailored to your specific needs."
            />
            <FeatureCard
              icon={<Moon className="h-10 w-10 text-indigo-400" />}
              title="Sleep Improvement"
              description="Get personalized sleep recommendations and wind-down routines based on your habits and preferences."
            />
          </div>
        </section>
      </main>

      <footer className="bg-slate-100 dark:bg-slate-900 py-12">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-4 md:mb-0">
              <Brain className="h-6 w-6 text-teal-500" />
              <span className="text-xl font-bold bg-gradient-to-r from-teal-500 to-sky-500 bg-clip-text text-transparent">
                MindFuel
              </span>
            </div>
            <div className="flex gap-8 text-slate-600 dark:text-slate-300">
              <Link href="/about" className="hover:text-teal-500 transition-colors">
                About
              </Link>
              <Link href="/privacy" className="hover:text-teal-500 transition-colors">
                Privacy
              </Link>
              <Link href="/terms" className="hover:text-teal-500 transition-colors">
                Terms
              </Link>
              <Link href="/contact" className="hover:text-teal-500 transition-colors">
                Contact
              </Link>
            </div>
          </div>
          <div className="mt-8 text-center text-slate-500 dark:text-slate-400">
            © {new Date().getFullYear()} MindFuel. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode
  title: string
  description: string
}) {
  return (
    <div className="bg-white dark:bg-slate-800 p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow border border-slate-100 dark:border-slate-700 group hover:border-teal-100 dark:hover:border-teal-900">
      <div className="mb-4 transform group-hover:scale-110 transition-transform">{icon}</div>
      <h3 className="text-xl font-semibold mb-3 text-slate-800 dark:text-white">{title}</h3>
      <p className="text-slate-600 dark:text-slate-300">{description}</p>
    </div>
  )
}
