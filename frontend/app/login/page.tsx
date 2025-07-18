import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain } from "lucide-react"

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-sky-50 to-white dark:from-slate-950 dark:to-slate-900 flex flex-col items-center justify-center p-4">
      <Link href="/" className="flex items-center gap-2 mb-8">
        <Brain className="h-8 w-8 text-teal-500" />
        <span className="text-2xl font-bold bg-gradient-to-r from-teal-500 to-sky-500 bg-clip-text text-transparent">
          MindFuel
        </span>
      </Link>

      <Card className="w-full max-w-md border-slate-200 dark:border-slate-800">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold text-center">Welcome back</CardTitle>
          <CardDescription className="text-center">Enter your credentials to access your account</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input id="email" type="email" placeholder="hello@example.com" />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="password">Password</Label>
              <Link href="/forgot-password" className="text-sm text-teal-500 hover:text-teal-600">
                Forgot password?
              </Link>
            </div>
            <Input id="password" type="password" />
          </div>
          <Button className="w-full bg-gradient-to-r from-teal-500 to-sky-500">Sign In</Button>
        </CardContent>
        <CardFooter className="flex flex-col space-y-4">
          <div className="text-sm text-slate-500 dark:text-slate-400 text-center">
            Don't have an account?{" "}
            <Link href="/register" className="text-teal-500 hover:text-teal-600">
              Sign up
            </Link>
          </div>
        </CardFooter>
      </Card>
    </div>
  )
}
