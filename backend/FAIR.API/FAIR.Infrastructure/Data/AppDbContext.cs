using FAIR.Domain.Entities;
using FAIR.Domain.Entities.Chat;
using FAIR.Domain.Entities.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;


namespace FAIR.Infrastructure.Data
{
    public class AppDbContext(DbContextOptions<AppDbContext> options) :IdentityDbContext<AppUser>(options)
    {
      //public DbSet<AppUser> Users { get; set; }
        public DbSet<Report> Reports { get; set; }
        public DbSet<Player> Players { get; set; }
        public DbSet<Coach> Coaches { get; set; }
        public DbSet<RefreshToken> RefreshToken { get; set; }
        public DbSet<Message> Messages { get; set; }

    }
}
